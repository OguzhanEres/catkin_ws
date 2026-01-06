# PX4 SITL + MAVROS Sorun Giderme Kılavuzu

## 1. QGroundControl Bağlantı Sorunu

### Mevcut Port Yapısı (PX4 SITL px4_instance=0):
| Port  | Kullanım                    | Bağlantı Yönü      |
|-------|-----------------------------|--------------------|
| 14580 | Offboard/API (PX4 dinliyor) | MAVROS → PX4       |
| 14540 | MAVROS local port           | PX4 → MAVROS       |
| 18570 | **GCS Link** (PX4 dinliyor) | QGC → PX4          |
| 14280 | Onboard Payload             | Kamera/Payload     |
| 13030 | Gimbal                      | Gimbal kontrol     |

### ❌ Sorun
QGroundControl varsayılan olarak **14550** portuna bağlanmaya çalışıyor, ama PX4 SITL GCS linkini **18570** portunda açıyor.

### ✅ Çözüm A: QGC Ayarlarını Değiştir (Önerilen)
1. QGroundControl'ü aç
2. **Application Settings** (dişli ikonu) → **Comm Links**
3. **UDP** linkini düzenle veya yeni ekle:
   - Type: **UDP**
   - Listening Port: **18570**
   - Target Host: `127.0.0.1`
   - Target Port: `18570`
4. **Connect** butonuna bas

### ✅ Çözüm B: PX4 GCS Portunu 14550'ye Çevir
PX4 çalışırken pxh shell'de:
```bash
# Mevcut 18570 linkini kapat (gerekirse)
mavlink stop-all

# 14550 portunda yeni GCS linki aç
mavlink start -x -u 14550 -r 4000000 -f
mavlink stream -r 50 -s POSITION_TARGET_LOCAL_NED -u 14550
mavlink stream -r 50 -s LOCAL_POSITION_NED -u 14550
mavlink stream -r 50 -s GLOBAL_POSITION_INT -u 14550
mavlink stream -r 50 -s ATTITUDE -u 14550
mavlink stream -r 20 -s RC_CHANNELS -u 14550
```

### ✅ Çözüm C: socat ile Port Yönlendirme
```bash
# Terminal'de çalıştır (arka planda)
socat UDP-LISTEN:14550,fork UDP:127.0.0.1:18570 &
```

---

## 2. Failsafe Tetiklenmesi (RC/Datalink/Offboard)

### Belirtiler
- Motorlar arm olduktan sonra kısılıyor
- AUTO.RTL veya AUTO.LAND'e düşüyor
- `[WARN] Failsafe: no RC/no datalink/no offboard` mesajı

### ✅ Çözüm: PX4 Parametrelerini Ayarla
PX4 pxh shell'de (veya QGC Parameter Editor'da):

```bash
# RC failsafe'i devre dışı bırak (SITL için)
param set COM_RC_IN_MODE 1
param set COM_RCL_EXCEPT 4
param set NAV_RCL_ACT 0

# Datalink failsafe'i devre dışı bırak
param set NAV_DLL_ACT 0
param set COM_DL_LOSS_T 300

# Offboard failsafe'i devre dışı bırak
param set COM_OBL_ACT 0

# RC loss timeout'u uzat
param set COM_RC_LOSS_T 300

# Low battery failsafe'i devre dışı bırak (simülasyon için)
param set COM_LOW_BAT_ACT 0

# Parametreleri kaydet
param save
```

### GCS Heartbeat Sağlama (Kritik!)
Failsafe'in kalıcı olarak kalkması için PX4'ün bir GCS'den heartbeat alması gerekir:

**Seçenek 1:** QGC bağla (yukarıdaki yöntemlerle)

**Seçenek 2:** MAVROS'un gcs_url parametresini ayarla:
```bash
roslaunch ardupilot_city_sim mavros_connect.launch \
  fcu_url:=udp://:14540@127.0.0.1:14580 \
  gcs_url:=udp://@127.0.0.1:18570
```

**Seçenek 3:** Dummy GCS heartbeat scripti:
```python
#!/usr/bin/env python3
# dummy_gcs_heartbeat.py
import socket
import struct
import time

def mavlink_heartbeat():
    """MAVLink 2.0 HEARTBEAT (system_id=255, component_id=190 = GCS)"""
    # Header: magic, len, incompat_flags, compat_flags, seq, sysid, compid, msgid(3bytes)
    magic = 0xFD  # MAVLink 2
    payload_len = 9
    incompat = 0
    compat = 0
    seq = 0
    sysid = 255  # GCS
    compid = 190  # MAV_COMP_ID_MISSIONPLANNER
    msgid = 0  # HEARTBEAT
    
    # Payload: type, autopilot, base_mode, custom_mode, system_status
    payload = struct.pack('<BBBIB',
        6,  # MAV_TYPE_GCS
        8,  # MAV_AUTOPILOT_INVALID
        0,  # base_mode
        0,  # custom_mode
        0   # MAV_STATE_UNINIT
    )
    
    return struct.pack('<BBBBBBBH', magic, payload_len, incompat, compat, seq, sysid, compid, msgid) + payload

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
target = ('127.0.0.1', 18570)

print("Sending GCS heartbeat to", target)
while True:
    sock.sendto(mavlink_heartbeat(), target)
    time.sleep(1)
```

---

## 3. OFFBOARD Modu Tutmuyor

### Belirtiler
- OFFBOARD'a geçiyor ama hemen POSCTL veya AUTO.LOITER'a düşüyor
- Setpoint akışı var ama mod değişiyor

### ✅ Çözüm
1. **Setpoint akışı kesintisiz olmalı** (>10 Hz):
   ```bash
   rostopic hz /mavros/setpoint_position/local
   # En az 10 Hz olmalı
   ```

2. **OFFBOARD'a geçmeden önce 1-2 saniye prestream** yapılmalı
   (Bu zaten `offboard_prestream_sec` parametresiyle yapılıyor)

3. **EKF pozisyon valid olmalı**:
   ```bash
   # PX4 pxh'de
   listener vehicle_local_position
   # xy_valid ve z_valid = true olmalı
   ```

4. **Vision bridge'in çalıştığını doğrula**:
   ```bash
   rostopic echo -n 1 /mavros/vision_pose/pose
   rostopic hz /mavros/vision_pose/pose
   # En az 20 Hz olmalı
   ```

---

## 4. Log Temizliği

### ~/.ros/log Uyarısı
```bash
# ROS loglarını temizle
rosclean purge -y

# Gazebo loglarını temizle
rm -rf ~/.gazebo/log/*

# PX4 loglarını temizle
rm -rf ~/PX4-Autopilot/build/px4_sitl_default/rootfs/log/*
```

---

## 5. Metrik/Model Kayıt Sorunu

Eğitim Ctrl+C ile kesildiğinde metrikler kayboluyorsa, trainer node'a shutdown handler eklenmeli:

```python
# ppo_lstm_trainer_node.py içinde __init__'e ekle:
import atexit
import signal

def _save_on_exit():
    if self.model is not None and self.checkpoint_path:
        torch.save(self.model.state_dict(), self.checkpoint_path)
        rospy.loginfo("Emergency checkpoint saved")

atexit.register(_save_on_exit)
signal.signal(signal.SIGINT, lambda s, f: (_save_on_exit(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (_save_on_exit(), sys.exit(0)))
```

---

## 6. Hızlı Test Komutları

```bash
# Port kullanımını kontrol et
lsof -i UDP -P -n | grep -E '145|185'

# PX4 MAVLink durumunu kontrol et (pxh'de)
mavlink status

# MAVROS bağlantı durumu
rostopic echo -n 1 /mavros/state

# Setpoint akış hızı
rostopic hz /mavros/setpoint_position/local

# EKF durumu (pxh'de)
listener estimator_status
listener vehicle_local_position
```

---

## 7. Önerilen Başlatma Sırası

1. **Gazebo** (Terminal A)
2. **PX4 SITL** (Terminal B) - Parametreler otomatik ayarlanır
3. **QGC'yi bağla** (18570 portuna)
4. **MAVROS** (Terminal C)
5. **DRL Training** (Terminal D)

Bu sıra GCS heartbeat'in MAVROS'tan önce gelmesini sağlar.
