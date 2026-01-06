# Proje Çalıştırma Kılavuzu (ROS Noetic + Gazebo 11 + PX4 SITL + DRL PPO+LSTM)

Bu doküman, şehir simülasyonu (Gazebo 11) + PX4 SITL + MAVROS + DRL (PPO+LSTM + IFDS) sistemini uçtan uca çalıştırmak için gerekli komutları ve kullanım adımlarını içerir.

## 0) Ön Koşullar

- Ubuntu 20.04 + ROS Noetic + Gazebo 11
- `catkin_ws` çalışma alanı: `/home/oguz/Desktop/catkin_ws`
- PX4 SITL: `/home/oguz/PX4-Autopilot`
- PX4 Gazebo pluginleri: `/home/oguz/PX4-Autopilot/build/px4_sitl_default/build_gazebo` (build edilmiş olmalı)

Bu repo/workspace içinde iki ROS paketi vardır:
- `ardupilot_city_sim` (şehir world + model/sensörler + GUI)
- `drl` (sensor vectorizer + PPO+LSTM trainer + IFDS + setpoint kontrol)

## 1) Build (Bir Kez)

```bash
source /opt/ros/noetic/setup.bash
cd ~/Desktop/catkin_ws
catkin_make
source ~/Desktop/catkin_ws/devel/setup.bash
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
  - `/mavros/setpoint_position/local` yayınlayarak OFFBOARD modda setpoint sürer
  - hedefe ulaşınca `/agent/wp_reached` yayınlar

## 3) Terminal ile Çalıştırma (Önerilen Debug Akışı)

### Terminal A — Gazebo (şehir world)
```bash
source /opt/ros/noetic/setup.bash
source ~/Desktop/catkin_ws/devel/setup.bash
roslaunch ardupilot_city_sim city_sim_gazebo.launch
```

### Terminal B — PX4 SITL (Sadece SITL, Gazebo başlatmadan)
```bash
cd ~/PX4-Autopilot
export PX4_SIM_MODEL=iris
export PX4_SIM_HOST_ADDR=127.0.0.1
./build/px4_sitl_default/bin/px4 ./build/px4_sitl_default/etc \
  -s etc/init.d-posix/rcS -t ./test_data
```

> Not: Gazebo zaten Terminal A’da çalışıyor olmalı. Bu komut sadece PX4 SITL başlatır.

### Terminal C — MAVROS bağlantısı
```bash
source /opt/ros/noetic/setup.bash
source ~/Desktop/catkin_ws/devel/setup.bash
roslaunch ardupilot_city_sim mavros_connect.launch fcu_url:=udp://:14540@127.0.0.1:14580
```

### Terminal D — DRL Training (tam DRL loop)
```bash
source /opt/ros/noetic/setup.bash
source ~/Desktop/catkin_ws/devel/setup.bash
roslaunch drl drl_train.launch mode:=exploration epochs:=10 \
  step_size:=1.0 takeoff_alt:=1.0 min_alt_for_control:=0.0 \
  reset_world:=true reboot_wait:=30
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
ls -l ~/Desktop/catkin_ws/src/drl/models/ppo_lstm_checkpoint.pt
stat ~/Desktop/catkin_ws/src/drl/models/ppo_lstm_checkpoint.pt
```

Drone durumu:
```bash
rostopic echo -n 1 /mavros/state
rostopic echo -n 1 /mavros/local_position/pose
```

## 4) GUI ile Çalıştırma

GUI, Gazebo/PX4/MAVROS/training launch işlemlerini butonlarla tetiklemek için vardır.

```bash
source /opt/ros/noetic/setup.bash
source ~/Desktop/catkin_ws/devel/setup.bash
rosrun ardupilot_city_sim sim_control_gui.py
```

Notlar:
- Training butonları sadece `roslaunch drl drl_train.launch ...` tetikler.
- Debug için terminal akışı daha nettir (bu README’de önerilen yol).

## 5) Kapatma / Temizlik

### ROS launch’ları durdurma
Her launch terminalinde `Ctrl+C`.

### PX4 süreçlerini kapatma
```bash
pkill -f px4
pkill -f gazebo
pkill -f gzserver
pkill -f gzclient
```

### Mission temizleme (MAVROS açıkken)
```bash
rosservice call /mavros/mission/clear "{}"
```

## 6) Sık Görülen Sorunlar ve Çözümler

### 6.1 “OFFBOARD’a geçmiyor / AUTO.RTL’e düşüyor”
Genellikle setpoint akışı kesildiğinde veya RC failsafe tetiklendiğinde olur.
- `/mavros/setpoint_position/local` akışı >10 Hz olmalı.
- PX4 terminalinde şu paramlar eğitim için önerilir:
  - `COM_RCL_EXCEPT=4`
  - `NAV_RCL_ACT=0`
  - `COM_OBL_ACT=0`
  - `COM_LOW_BAT_ACT=0`
- `/mavros/statustext/recv` ile failsafe mesajlarını kontrol edin:
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

### 6.4 `/mavros/local_position/pose` akmıyor (EKF xy_valid=false)
PX4, güvenli yaw/pozisyon olmadığında LOCAL_POSITION_NED yayınlamaz → OFFBOARD/arming reddedilir. Gazebo pozunu EKF’ye beslemek için `drl_train.launch` içinde varsayılan olarak `gazebo_vision_bridge` açıldı. Bu node `/gazebo/model_states` → `/mavros/vision_pose/pose` (ve hız) yayınlayarak EKF’nin xy_valid/yaw referansını sağlamlaştırır.
- Çalıştığını doğrulamak için: `rostopic echo -n 1 /mavros/local_position/pose` ve PX4 terminalinde `listener vehicle_local_position`.
- Donanım testlerinde gerekirse kapatın: `roslaunch drl drl_train.launch use_vision_bridge:=false ...`.

## 7) Önemli Dosyalar

- Şehir world: `~/Desktop/catkin_ws/src/ardupilot_city_sim/worlds/city_sim.world`
- Drone model (plugin + sensör): `~/Desktop/catkin_ws/src/ardupilot_city_sim/models/iris_with_lidar_camera_px4/`
- DRL launch: `~/Desktop/catkin_ws/src/drl/launch/drl_train.launch`
- DRL checkpoint: `~/Desktop/catkin_ws/src/drl/models/ppo_lstm_checkpoint.pt`
source /opt/ros/noetic/setup.bash
source ~/Desktop/catkin_ws/devel/setup.bash
roslaunch ardupilot_city_sim city_sim_gazebo.launch


cd ~/PX4-Autopilot
export PX4_SIM_MODEL=iris
export PX4_SIM_HOST_ADDR=127.0.0.1
./build/px4_sitl_default/bin/px4 ./build/px4_sitl_default/etc \
  -s etc/init.d-posix/rcS -t ./test_data


source /opt/ros/noetic/setup.bash
source ~/Desktop/catkin_ws/devel/setup.bash
roslaunch ardupilot_city_sim mavros_connect.launch fcu_url:=udp://:14540@127.0.0.1:14580


source /opt/ros/noetic/setup.bash
source ~/Desktop/catkin_ws/devel/setup.bash
roslaunch drl drl_train.launch mode:=exploration epochs:=10 \
  step_size:=1.0 takeoff_alt:=1.0 min_alt_for_control:=0.0 \
  reset_world:=true reboot_wait:=30 reboot_each_episode:=true
# yumuşatma/settle parametreleri eklemek istersen:
# max_xy_speed:=1.0 max_z_speed:=0.5 target_settle_time:=1.0 require_settle_for_new_target:=true
