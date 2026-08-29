"""
servo_controller.py
--------------------
Bo dieu khien servo pan/tilt: KHONG dung PID/lam muot toc do -- chi di
chuyen tung BUOC CO DINH NHO (khong tang/giam toc), nghi (cooldown) sau moi
buoc de cho camera vat ly xoay xong + AI kip xu ly ra frame moi, roi moi
danh gia buoc tiep theo. Uu tien do TIN CAY va ON DINH (cham nhung chac)
hon la toc do bam theo.

  1) Sai so (error_x, error_y) den tu AIPipeline (ai_pipeline.py) -- tam O
     VUONG XANH (bbox) cua nguoi DA DANG KY dang duoc khoa, so voi tam
     khung hinh.

  2) TRANSPORT: gui lenh qua WebSocket dung chung voi 2 servo cua
     (operation/door_ws_server.py -> esp32_servo.ino).

  3) SCAN MODE: ngoai che do bam muc tieu (update()), servo con co the
     duoc yeu cau chu dong "quet" pan+tilt qua lai trong mot dai goc chi
     dinh (start_scan()/tick_scan()/stop_scan()) khi khong con error do
     duoc de bam theo.

ServoController khong quan tam muc tieu la ai / duoc tim thay bang cach
nao (FACE / BODY / BODY_SHAPE) -- no chi nhan (error_x, error_y) moi frame
va tinh goc moi, roi gui qua door_ws.send_pan_tilt(pan, tilt).
"""

import time

class ServoController:
    def __init__(self, config: dict, door_ws=None):
        """
        door_ws: an operation.door_ws_server.DoorWebSocketServer instance
            (the SAME one app_dashboard.py uses for the door servos) --
            pan/tilt commands are sent via door_ws.send_pan_tilt(pan, tilt)
            over that shared WebSocket connection. If None (e.g. running
            under app_operation.py's standalone CLI dashboard, which has
            no DoorWebSocketServer), the controller still computes angles
            normally but simply doesn't send anything -- same graceful
            "simulate" behavior the old Serial-based version had when no
            hardware was plugged in.
        """
        c = config["control"]

        # Buoc di chuyen CO DINH moi lan (do). KHONG tang/giam toc -- luon
        # dung dung 1 gia tri nay (hoac nho hon, neu gan toi bien o xanh).
        # TACH RIENG pan/tilt vi tilt (ngan len/xuong) thuong can CHAM HON
        # pan -- de nguoi dung tinh chinh doc lap trong config.py.
        # Muon servo cham hon -> giam so nay xuong (vd 1). Muon nhanh hon
        # (nhung de mat on dinh hon) -> tang len.
        self.max_step_pan = c.get("max_step_per_frame", 3)
        self.max_step_tilt = c.get("max_step_per_frame_tilt", 1)

        # Thoi gian (giay) "khung" sau MOI lan gui lenh di chuyen thuc su
        # -- trong khoang nay, du co frame moi cung KHONG tinh/gui lenh
        # tiep, cho camera vat ly xoay xong + AI kip xu ly frame moi phan
        # anh dung vi tri, roi moi danh gia lai. Day la phan quan trong
        # nhat giup servo "cham ma chac", khong dao dong qua lai.
        self.post_move_settle_sec = c.get("post_move_settle_sec", 0.25)
        self._last_move_time = 0.0

        self.dead_zone = c["dead_zone_px"]
        # San toi thieu cho dead-zone DONG (theo o xanh thuc te). Neu o
        # xanh qua nho (nguoi o xa, box chi vai chuc px) ma bat servo dung
        # chinh xac tuyet doi trong tung px se khien no rung/khong bao gio
        # thuc su "on dinh". min_dead_zone_px dam bao luon co mot vung dung
        # toi thieu du rong du o xanh nho co nao.
        self.min_dead_zone = c.get("min_dead_zone_px", 25)
        # Bien do "cham" (px) -- CONG THEM vao ranh gioi o xanh, de servo
        # dung ngay khi tam khung hinh CHAM TOI MEP o xanh (hoac gan mep),
        # KHONG can di vao han ben trong moi dung. Tang so nay len neu
        # muon dung som hon nua (tu xa mep da coi la "cham" duoc roi);
        # dat ve 0 neu muon dung dung khi thuc su lot vao ben trong.
        self.touch_margin = c.get("touch_margin_px", 20)

        # --- KHOA (hysteresis) sau khi da dung -- day la phan giai quyet
        # cam bam theo dung tam ban chat: neu khong co, MOI frame deu tu
        # tinh lai so voi CHINH XAC ranh gioi o xanh hien tai, nen chi can
        # o xanh xe dich nhe (rung camera, nguoi lac lu nhe...) la lai
        # nhich tiep -> cam giac "luc nao cung co di vao tam". Co khoa:
        # mot khi DA dung (settled) o 1 truc, truc do se DUNG YEN HAN cho
        # toi khi sai so vuot qua nguong NHA KHOA (release_margin_px,
        # RONG HON nhieu so voi luc dung) moi tinh la "roi khoi o xanh
        # that su" va bat dau di chuyen tiep. Cang tang release_margin_px
        # cang it bi "nhich" theo nhung rung/lac nho.
        self.release_margin = c.get("release_margin_px", 90)
        self._settled_x = True
        self._settled_y = True

        self.pan_min, self.pan_max = c["pan_min"], c["pan_max"]
        self.tilt_min, self.tilt_max = c["tilt_min"], c["tilt_max"]
        self.invert_pan = c.get("invert_pan", False)
        self.invert_tilt = c.get("invert_tilt", False)

        self.pan_angle = c["pan_center"]
        self.tilt_angle = c["tilt_center"]

        self.send_interval = c.get("send_interval_sec", 0.05)
        self._last_send = 0.0

        # --- Che do quet chu dong (khong co error do) ---
        # Dung khi ban giao muc tieu sang phong khac va phong nguon can
        # chu dong xoay di "tim" thay vi dung yen cho gap lai.
        self.scanning = False
        self._scan_pan_dir = 1
        self._scan_tilt_dir = 1
        self._scan_cfg = None

        self.door_ws = door_ws
        self.simulate = door_ws is None
        if self.simulate:
            print(
                "[servo_controller] Khong co door_ws (WebSocket server) duoc truyen vao -- "
                "chay o che do MO PHONG (van tinh goc, khong gui lenh xuong ESP32)."
            )
        else:
            print("[servo_controller] Se gui lenh pan/tilt qua WebSocket dung chung voi cua (esp32_servo.ino).")

    def _clamp(self, value, lo, hi):
        return max(lo, min(hi, value))

    def reset_integral(self):
        """Goi khi mat khoa muc tieu / huy khoa / an toan ve tam. Giu lai
        ten ham de tuong thich voi cac noi khac dang goi no (ai_pipeline.py).
        Reset ca dong ho cooldown lan trang thai khoa (settled) moi truc,
        de lan bam muc tieu tiep theo bat dau "sach", khong dinh trang
        thai khoa/nghi con sot lai tu lan truoc."""
        self._last_move_time = 0.0
        self._settled_x = True
        self._settled_y = True

    def _axis_step(self, error, dead_zone, settled, max_step):
        """Tinh buoc di chuyen CO DINH cho 1 truc (dung chung pan/tilt),
        CO KHOA (hysteresis). Tra ve (step, settled_moi).

        max_step: buoc toi da (do) RIENG cho truc nay (pan va tilt co the
        khac nhau -- vd tilt cham hon pan).

        KHONG PID, KHONG lam muot, KHONG tang/giam toc. Nhung CO nho trang
        thai da-dung-hay-chua (settled) de tao 2 nguong khac nhau:
          - Truc DANG dung (settled=True): chi can roi ra ngoai nguong
            NHA KHOA rong (dead_zone + self.release_margin) moi tinh la
            "thoat khoi o xanh that su" va bat dau di chuyen lai. Con nam
            trong nguong nay (du hoi lech khoi mep o xanh mot chut do
            rung/lac nhe) thi VAN DUNG YEN, khong nhich.
          - Truc CHUA dung (settled=False, dang di chuyen toi): dung
            nguong CHAT dead_zone (co touch_margin) nhu binh thuong de
            biet khi nao da "cham" toi o xanh va duoc phep khoa lai.
        Nho vay servo se KHONG con lien tuc "co di vao tam" moi khi o
        xanh xe dich nhe sau khi da dung -- chi thuc su di chuyen lai khi
        muc tieu roi hang han khoi vung an toan.

        LUU Y (buoc 5 do nhung KHONG duoc nhay qua luon o xanh): step
        DUNG BANG max_step (vd 5 do) MOI KHI con xa, nhung o BUOC CUOI
        CUNG (khi phan con lai toi bien o xanh nho hon max_step) thi CHI
        di dung phan con lai do -- de servo luon "dap" dung vao trong/mep
        o xanh thay vi nhay vot qua ben kia. Neu khong co gioi han nay,
        buoc 5 do co the nhay het qua ca o xanh (neu 5 do tuong ung nhieu
        pixel hon be rong o xanh) khien khong bao gio co 1 frame nao ghi
        nhan duoc "da vao trong" de dung -- gay dao dong qua lai mai
        khong dut, dung la trieu chung ban gap phai.
        """
        threshold = dead_zone + self.release_margin if settled else dead_zone

        if abs(error) <= threshold:
            return 0.0, True

        remaining_to_edge = abs(error) - dead_zone
        step = min(max_step, remaining_to_edge)
        return (step if error > 0 else -step), False

    def update(self, error_x, error_y, box_half_w=None, box_half_h=None):
        """
        Tinh goc pan/tilt moi: di chuyen BUOC CO DINH NHO (self.max_step
        do) ve phia muc tieu, roi NGHI (post_move_settle_sec giay) truoc
        khi danh gia buoc tiep theo. Khong PID, khong tang/giam toc --
        uu tien "cham ma chac", tranh dao dong/chay lo. Co KHOA (xem
        _axis_step) de khong lien tuc "co di vao tam" khi da dung roi.

        box_half_w / box_half_h: nua-be-rong / nua-chieu-cao (px) cua O
        VUONG XANH thuc te (bbox nguoi dang khoa) trong frame hien tai, do
        ai_pipeline.py truyen xuong. Neu co, servo se DUNG NGAY KHI TAM
        KHUNG HINH (crosshair) LOT VAO BEN TRONG O XANH DO -- dung nhu
        nguoi dung nhin thay tren dashboard. Neu khong truyen (None, vd
        dang scan/preempt), dung nguong tinh self.dead_zone nhu cu.

        Tra ve (pan_angle, tilt_angle).
        """
        # --- COOLDOWN sau lan di chuyen truoc: neu vua gui lenh di chuyen
        # cach day chua du self.post_move_settle_sec giay, BO QUA hoan toan
        # frame nay (khong tinh step, khong doi angle) -- cho camera vat ly
        # xoay xong + AI kip xu ly ra frame moi phan anh DUNG vi tri hien
        # tai, roi moi danh gia tiep. Day la phan chinh giup servo "cham ma
        # chac": khong bao gio ra lenh moi khi con chua chac chan anh da
        # cap nhat theo lan xoay truoc.
        now = time.time()
        if (now - self._last_move_time) < self.post_move_settle_sec:
            return self.pan_angle, self.tilt_angle

        # Dead-zone DONG theo o xanh that, co san toi thieu (min_dead_zone)
        # de tranh o xanh qua nho (nguoi o xa) lam servo rung/khong bao gio
        # thuc su on dinh, CONG THEM touch_margin de chi can CHAM MEP o
        # xanh la dung, khong can di vao han ben trong.
        dead_zone_x = (max(box_half_w, self.min_dead_zone) if box_half_w is not None else self.dead_zone) + self.touch_margin
        dead_zone_y = (max(box_half_h, self.min_dead_zone) if box_half_h is not None else self.dead_zone) + self.touch_margin

        step_pan, self._settled_x = self._axis_step(error_x, dead_zone_x, self._settled_x, self.max_step_pan)
        step_tilt, self._settled_y = self._axis_step(error_y, dead_zone_y, self._settled_y, self.max_step_tilt)

        if step_pan == 0.0 and step_tilt == 0.0:
            # Tam khung hinh da nam gon trong o xanh (hoac van con trong
            # vung khoa) tren ca 2 truc -> dung han, khong gui gi them.
            self._send()
            return self.pan_angle, self.tilt_angle

        if self.invert_pan:
            step_pan = -step_pan
        if self.invert_tilt:
            step_tilt = -step_tilt

        self.pan_angle = self._clamp(self.pan_angle + step_pan, self.pan_min, self.pan_max)
        self.tilt_angle = self._clamp(self.tilt_angle - step_tilt, self.tilt_min, self.tilt_max)

        # Vua di chuyen that -> bat dau lai dong ho cooldown.
        self._last_move_time = now

        self._send()
        return self.pan_angle, self.tilt_angle

    def go_to_center(self, config):
        """Co che an toan khi mat muc tieu: dua servo ve vi tri mac dinh.
        Cung dung de KET THUC che do quet chu dong (vd khi phong dich da
        tim ra nguoi vua ban giao)."""
        self.stop_scan()
        self.reset_integral()
        self.pan_angle = config["control"]["pan_center"]
        self.tilt_angle = config["control"]["tilt_center"]
        self._send(force=True)

    def preempt_to_angle(self, pan_angle, tilt_angle=None):
        """
        Lenh 'don dau' (handoff preempt): ep servo quay NGAY toi pan_angle
        chi dinh, KHONG qua PID/ramp-up thong thuong -- vi luc nay chua he
        co sai so do (error_x/error_y) nao ca, ta dang CHU DONG doan truoc
        vi tri doi tuong SAP xuat hien dua tren tin hieu tu phong khac, chu
        khong phai dang bam theo detection cua chinh phong nay.

        Reset luon bo dieu khien PID de khong bi nhieu boi sai so/tich luy
        cu con sot lai tu lan bam muc tieu truoc do, va huy che do quet chu
        dong neu dang bat (uu tien lenh don dau tuc thi hon quet).
        """
        self.stop_scan()
        self.reset_integral()
        self.pan_angle = self._clamp(pan_angle, self.pan_min, self.pan_max)
        if tilt_angle is not None:
            self.tilt_angle = self._clamp(tilt_angle, self.tilt_min, self.tilt_max)
        self._send(force=True)

    # -------------------------------------------------------------------
    # Che do quet chu dong (active scan) -- dung khi ban giao muc tieu
    # sang phong khac va can chu dong xoay tim thay vi dung yen cho.
    # -------------------------------------------------------------------

    def start_scan(self, pan_min, pan_max, tilt_min, tilt_max, step_deg=3, tilt_center=None):
        """
        Bat dau quet chu dong CHI PAN qua lai trong dai [pan_min, pan_max]
        (khong dung PID, vi khong co error do duoc -- day la "mo" tim,
        khac han bam muc tieu binh thuong). TILT duoc dua ve co dinh o
        tilt_center (mac dinh 90 do) va GIU NGUYEN suot qua trinh quet --
        khong quet doc theo tilt nua, theo yeu cau thuc te (quet ca tilt
        tu 40->160 lam camera "nguoc len" qua cao, khong huu ich).

        Dung khi: muc tieu vua duoc ban giao sang phong khac (vd Phong 2 ->
        Phong 1) va da qua thoi gian cho (config.HANDOFF_WAIT_BEFORE_SCAN_SEC)
        ma phong dich van chua tu tim ra nguoi do.

        pan_min/pan_max: thuong la bien logic cua phong (vd
            HANDOFF_CONFIG["cam2"]["pan_left_boundary"/"pan_right_boundary"]),
            khong nhat thiet trung voi gioi han co khi pan_min/pan_max cua
            servo.
        tilt_min/tilt_max: GIOI HAN AN TOAN cho tilt trong che do quet (vd
            config.HANDOFF_SCAN_TILT_MIN/MAX = 40/160) -- chi dung de KEP
            tilt_center vao trong khoang nay cho an toan co khi, KHONG con
            dung de quet doc nua.
        tilt_center: goc tilt co dinh khi quet. None -> dung tilt_center
            cua SERVO_CONFIG (thuong la 90 do).
        """
        self.reset_integral()
        self.scanning = True
        if tilt_center is None:
            tilt_center = (self.tilt_min + self.tilt_max) / 2.0
        tilt_center = self._clamp(tilt_center, tilt_min, tilt_max)
        self._scan_cfg = {
            "pan_min": pan_min,
            "pan_max": pan_max,
            "tilt_center": tilt_center,
            "step": step_deg,
        }
        self._scan_pan_dir = 1
        # Dua tilt ve co dinh NGAY khi bat dau quet, gui lenh ngay lap tuc
        # thay vi doi den tick dau tien -- tranh tilt bi "treo" o goc cu
        # (vd 150-160 do neu vua bam muc tieu o bien tren truoc do).
        self.tilt_angle = self._clamp(tilt_center, self.tilt_min, self.tilt_max)
        self._send(force=True)

    def stop_scan(self):
        """Dung che do quet chu dong (goc hien tai giu nguyen, khong tu
        nhay ve center -- goi go_to_center() rieng neu muon ve mac dinh)."""
        self.scanning = False
        self._scan_cfg = None

    def tick_scan(self):
        """
        Goi dinh ky (vd moi HANDOFF_SCAN_TICK_INTERVAL_SEC giay, tu
        HandoffManager.check_pending() trong app_dashboard.py) de tien
        them 1 buoc quet PAN. Khong lam gi neu scanning=False.

        Pan di chuyen kieu "con thoi" (bat khi cham bien thi doi chieu).
        Tilt GIU NGUYEN o tilt_center da chot luc start_scan() -- khong
        con quet doc theo tilt.
        """
        if not self.scanning or self._scan_cfg is None:
            return
        c = self._scan_cfg

        self.pan_angle += self._scan_pan_dir * c["step"]
        if self.pan_angle >= c["pan_max"]:
            self.pan_angle = c["pan_max"]
            self._scan_pan_dir = -1
        elif self.pan_angle <= c["pan_min"]:
            self.pan_angle = c["pan_min"]
            self._scan_pan_dir = 1

        # Tilt co dinh -- khong tang/giam theo buoc quet nua.
        self.tilt_angle = c["tilt_center"]

        # Van kep trong gioi han co khi cua servo (pan_min/pan_max,
        # tilt_min/tilt_max) de khong bao gio vuot qua phan cung.
        self.pan_angle = self._clamp(self.pan_angle, self.pan_min, self.pan_max)
        self.tilt_angle = self._clamp(self.tilt_angle, self.tilt_min, self.tilt_max)

        self._send(force=True)

    def _send(self, force=False):
        now = time.time()
        if not force and (now - self._last_send) < self.send_interval:
            return
        self._last_send = now

        if self.simulate or self.door_ws is None:
            return  # no WebSocket server reference -- angles still computed, just not sent

        self.door_ws.send_pan_tilt(self.pan_angle, self.tilt_angle)
        # send_pan_tilt() is fire-and-forget and returns False if the
        # ESP32 isn't currently connected -- not fatal, PID state keeps
        # updating normally and delivery resumes the moment it reconnects
        # (esp32_servo.ino auto-retries every 3s).

    def close(self):
        """No persistent connection of our own to close -- door_ws (owned
        by app_dashboard.py) manages the actual WebSocket lifecycle.
        Kept for API compatibility with the old Serial-based version."""
        pass