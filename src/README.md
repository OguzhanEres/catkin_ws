# Proje Çalıştırma Kılavuzu (ROS Noetic + Gazebo 11 + ArduPilot SITL + DRL PPO+LSTM)

Bu doküman, şehir simülasyonu (Gazebo 11) + ArduPilot SITL + MAVROS + DRL (PPO+LSTM + IFDS) sistemini uçtan uca çalıştırmak için gerekli komutları ve kullanım adımlarını içerir.

## 0) Ön Koşullar

- Ubuntu 20.04 + ROS Noetic + Gazebo 11
- `catkin_ws` çalışma alanı: `/home/oguz/catkin_ws`
- ArduPilot SITL: `/home/oguz/ardupilot`
- ArduPilot Gazebo plugin: `/home/oguz/ardupilot_gazebo` (build edilmiş olmalı)

Bu repo/workspace içinde iki ROS paketi vardır:
- `ardupilot_city_sim` (şehir world + model/sensörler + GUI)
- `drl` (sensor vectorizer + PPO+LSTM trainer + IFDS + setpoint kontrol)

## 1) Build (Bir Kez)

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## 2) Sistem Mimarisi (Kısa)

Sensör ve kontrol veri akışı:

1) Gazebo sensörleri
- LiDAR: `/scan`
- Kamera: `/front_camera/image_raw`

2) MAVROS telemetri
- IMU: `/mavros/imu/data`
- GPS: `/mavros/global_position/global`
- Local pose: `/mavros/local_position/pose`

3) DRL
- `sensor_vectorizer_node.py`:
  - `/scan`, `/front_camera/image_raw`, `/mavros/imu/data`, `/mavros/global_position/global` alır
  - `/agent/*_vec` vektörlerini üretir
- `ppo_lstm_trainer_node.py`:
  - obs vektörlerinden PPO+LSTM aksiyonu üretir
  - `/agent/route_raw` yayınlar
  - reward hesaplar, PPO update yapar, checkpoint yazar
- `ifds_smoother_node.py`:
  - `/agent/route_raw` → `/agent/route_smoothed`
- `setpoint_follower_node.py` (tam DRL loop kontrol katmanı):
  - `/agent/route_smoothed`’daki ilk hedefi alır
  - `/mavros/setpoint_position/local` yayınlayarak GUIDED modda setpoint sürer
  - hedefe ulaşınca `/agent/wp_reached` yayınlar

## 3) Terminal ile Çalıştırma (Önerilen Debug Akışı)

### Terminal A — Gazebo (şehir world)
```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch ardupilot_city_sim city_sim_gazebo.launch
```

### Terminal B — ArduPilot SITL (ArduCopter)
Önerilen: param override ile başlat (SITL için)
```bash
cd ~/ardupilot/ArduCopter
../Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris -I0 \
  --out=127.0.0.1:14550 \
  --add-param-file=$HOME/catkin_ws/src/ardupilot_city_sim/config/ardupilot_override.parm
```

> Not: Bu komut MAVProxy’yi de açar ve ArduCopter terminalini ayrı pencerede çalıştırır.

### Terminal C — MAVROS bağlantısı
```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch ardupilot_city_sim mavros_connect.launch
```

### Terminal D — DRL Training (tam DRL loop)
```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch drl drl_train.launch mode:=exploration epochs:=10000 step_size:=10.0 takeoff_alt:=2.0
```

Parametre notları:
- `step_size`: hedef adım büyüklüğü (çok büyük olursa agresif setpoint → dengesizlik; 5–15 arası başlayın)
- `takeoff_alt`: otomatik takeoff hedefi (genelde 2m)

### Eğitim Çıktıları (kontrol)

Epoch logları:
```bash
rostopic echo /rosout | grep "Epoch "
```

Checkpoint:
```bash
ls -l ~/catkin_ws/src/drl/models/ppo_lstm_checkpoint.pt
stat ~/catkin_ws/src/drl/models/ppo_lstm_checkpoint.pt
```

Drone durumu:
```bash
rostopic echo -n 1 /mavros/state
rostopic echo -n 1 /mavros/local_position/pose
```

## 4) GUI ile Çalıştırma

GUI, Gazebo/ArduPilot/MAVROS/training launch işlemlerini butonlarla tetiklemek için vardır.

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
rosrun ardupilot_city_sim sim_control_gui.py
```

Notlar:
- Training butonları sadece `roslaunch drl drl_train.launch ...` tetikler.
- Debug için terminal akışı daha nettir (bu README’de önerilen yol).

## 5) Kapatma / Temizlik

### ROS launch’ları durdurma
Her launch terminalinde `Ctrl+C`.

### ArduPilot/MAVProxy süreçlerini kapatma
```bash
pkill -f sim_vehicle.py
pkill -f mavproxy.py
pkill -f arducopter
```

### Mission temizleme (MAVROS açıkken)
```bash
rosservice call /mavros/mission/clear "{}"
```

## 6) Sık Görülen Sorunlar ve Çözümler

### 6.1 “Arming rejected / Takeoff rejected”
Genellikle EKF/GPS hazır değilken olur.
- SITL başladıktan sonra 10–20 sn bekleyin.
- `drl_train.launch` içinde **preflight gate** aktif: Trainer, `PreArm:`/`EKF` hataları varken arm/takeoff denemez.
- `/mavros/statustext/recv` ile PreArm mesajlarını kontrol edin:
```bash
rostopic echo /mavros/statustext/recv
```

### 6.2 “Crash: Disarming: AngErr=...”
Setpoint agresif veya EKF kararsız:
- `step_size` değerini düşürün (örn. 10.0 → 5.0)
- Sistem kararlı kalkış yapana kadar training’i başlatmayın

### 6.3 NED/ENU z işareti
Bazı koşullarda `/mavros/local_position/pose` z negatif görülebilir.
Kontrol katmanı bu durumu otomatik algılayıp güvenli hedef üretmeye çalışır.

## 7) Önemli Dosyalar

- Şehir world: `~/catkin_ws/src/ardupilot_city_sim/worlds/city_sim.world`
- Drone model (plugin + sensör): `~/catkin_ws/src/ardupilot_city_sim/models/iris_with_lidar_camera_ros/`
- ArduPilot param override: `~/catkin_ws/src/ardupilot_city_sim/config/ardupilot_override.parm`
- DRL launch: `~/catkin_ws/src/drl/launch/drl_train.launch`
- DRL checkpoint: `~/catkin_ws/src/drl/models/ppo_lstm_checkpoint.pt`
