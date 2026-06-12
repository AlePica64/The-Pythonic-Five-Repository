import socket
import sys
import time
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import glob

PI = 3.14159265359
data_size = 2**17

# ==============================================================================
# 1. PARAMETRI CONFIGURABILI
# ==============================================================================
TARGET_SPEED = 200
STEER_GAIN = 35
CENTERING_GAIN = 0.3          
BRAKE_THRESHOLD = 0.35
GEAR_SPEEDS = [0, 50, 80, 120, 150, 200]
ENABLE_TRACTION_CONTROL = True

SAFE_GENTLE_CORNER_SPEED = 150
SAFE_SHARP_CORNER_SPEED = 60  
TARGET_STRAIGHT_SPEED = 350   
CORNER_READING = 2.0

SLOW_DOWN_DISTANCE = 45       
STRAIGHT_DISTANCE = 71
BRAKING_INTENSITY = 0.3
STEERING_EFFECT = 1.8

# ==============================================================================
# 2. MODELLO KNN
# ==============================================================================
class KNNDriver:
    def __init__(self, k=15):
        self.k = k
        self.model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        self.scaler = StandardScaler()
        self.is_trained = False

    def train_model(self, file_list):
        dataframes = []
        colonne_necessarie = {'speedX', 'trackPos', 'angle', 'target_steer', 'target_accel', 'target_brake'}
        
        for file in file_list:
            try:
                df = pd.read_csv(file)
                df.columns = df.columns.str.strip()
                
                if colonne_necessarie.issubset(df.columns):
                    dataframes.append(df)
                else:
                    print(f"[SKIP] Ignoro '{file}': mancano le colonne necessarie.")
            except Exception as e:
                print(f"[ERRORE] Impossibile leggere {file}: {e}")

        if not dataframes:
            print("[ERRORE CRITICO] Nessun file valido trovato per l'addestramento!")
            return False

        dataset = pd.concat(dataframes, ignore_index=True)
        dataset = dataset[dataset['trackPos'].between(-1.10, 1.10)]

        X = dataset[['speedX', 'trackPos', 'angle']].values
        y = dataset[['target_steer', 'target_accel', 'target_brake']].values

        X_scaled = self.scaler.fit_transform(X)
        X_scaled[:, 1] = X_scaled[:, 1] * 2.0
        X_scaled[:, 2] = X_scaled[:, 2] * 3.5

        self.model.fit(X_scaled, y)
        self.is_trained = True
        print(f"Modello addestrato su {len(dataset)} record!")
        return True

    def predict(self, speed, trackPos, angle):
        if not self.is_trained:
            return [0.0, 0.0, 0.0]
        current_state = np.array([[speed, trackPos, angle]])
        current_state_scaled = self.scaler.transform(current_state)
        current_state_scaled[0, 1] = current_state_scaled[0, 1] * 2.0
        current_state_scaled[0, 2] = current_state_scaled[0, 2] * 3.5
        return self.model.predict(current_state_scaled)[0]


# ==============================================================================
# 3. FUNZIONI DI SUPPORTO 
# ==============================================================================
def clip(v, lo, hi):
    if v < lo:   return lo
    elif v > hi: return hi
    else:        return v

def get_min_sensor_data(S):
    left_sensors  = S.get('track', [200] * 19)[:9]
    right_sensors = S.get('track', [200] * 19)[10:]
    return min(min(left_sensors), min(right_sensors))

def is_corner(S, min_reading):
    if min_reading < CORNER_READING or S.get('track', [200] * 19)[9] < S.get('speedX', 0) * 0.65:
        return True
    return False

def is_straight(current_speed, forward_length):
    if current_speed >= (TARGET_SPEED - 5) and forward_length > STRAIGHT_DISTANCE:
        return True
    return False

def hold_acceleration(S, safe_speed):
    min_sensor_data = get_min_sensor_data(S)
    if is_corner(S, min_sensor_data) and S.get('speedX', 0) > safe_speed:
        return True
    return False

def slow_down(S):
    max_forwards_sensors = max(S.get('track', [200] * 19)[7:12])
    if max_forwards_sensors < S.get('speedX', 0) * 0.60:
        return True
    return False

def calculate_corner_speed(S):
    max_forwards_sensors = max(S.get('track', [200] * 19)[8:11])
    safe_speed = SAFE_GENTLE_CORNER_SPEED
    if max_forwards_sensors < SLOW_DOWN_DISTANCE:
        safe_speed = SAFE_SHARP_CORNER_SPEED
    return safe_speed

def calculate_steering(S):
    steer = (S.get('angle', 0) * STEER_GAIN / PI) - (S.get('trackPos', 0) * CENTERING_GAIN)
    track     = S.get('track', [200] * 19)
    speed_x   = S.get('speedX', 0)

    left_near  = sum(track[6:9])  / 3.0   
    right_near = sum(track[10:13]) / 3.0  
    left_far   = sum(track[3:6])  / 3.0   
    right_far  = sum(track[13:16]) / 3.0  

    near_imbalance = (left_near - right_near) / 200.0
    far_imbalance  = (left_far  - right_far)  / 200.0
    speed_gain = min(1.0, max(0.0, speed_x / 200.0))

    predictive_steer = -(near_imbalance * 0.4 + far_imbalance * 0.15) * speed_gain
    steer += predictive_steer

    if is_corner(S, get_min_sensor_data(S)):
        left_avg  = sum(track[:9])  / 8
        right_avg = sum(track[10:]) / 8
        bias = right_avg - left_avg
        if bias < 0: steer += 0.30
        elif bias > 0: steer -= 0.30

    return max(-1, min(1, steer))

def calculate_throttle(S, R):
    target_speed = TARGET_SPEED
    safe_speed   = calculate_corner_speed(S)

    if is_straight(S.get('speedX', 0), S.get('track', [200] * 19)[9]):
        target_speed = TARGET_STRAIGHT_SPEED

    if S.get('speedX', 0) < target_speed - (R.get('steer', 0) * STEERING_EFFECT):
        accel = min(1.0, R.get('accel', 0) + 0.5)
    else:
        accel = max(0.0, R.get('accel', 0) - 0.7)

    if hold_acceleration(S, safe_speed):
        accel = max(0.0, R.get('accel', 0) - 0.4)

    if S.get('speedX', 0) < 10:
        accel = 1.0

    return max(0.0, min(1.0, accel))

def apply_brakes(S):
    brake    = 0.0
    speed_x  = S.get('speedX', 0)
    angle    = S.get('angle',  0)
    track    = S.get('track', [200] * 19)

    if abs(angle) > BRAKE_THRESHOLD:
        brake = BRAKING_INTENSITY

    if slow_down(S):
        brake += 0.15

    front_dist = track[9] if len(track) > 9 else 200.0
    required_braking_dist = speed_x * 0.35  

    if front_dist < required_braking_dist and front_dist > 0:
        urgency = 1.0 - (front_dist / required_braking_dist)
        preventive_brake = urgency * 0.5
        brake = max(brake, preventive_brake)

    left_near  = min(track[6:9])  if len(track) >= 9  else 200.0
    right_near = min(track[10:13]) if len(track) >= 13 else 200.0
    lateral_imbalance = abs(left_near - right_near) / 200.0
    if lateral_imbalance > 0.3 and speed_x > 80:
        brake += lateral_imbalance * 0.25

    return min(1.0, brake)

def shift_gears(S):
    gear = 1
    for i, speed in enumerate(GEAR_SPEEDS):
        if S.get('speedX', 0) > speed:
            gear = i + 1
    return min(gear, 6)

def traction_control(S, accel):
    if ENABLE_TRACTION_CONTROL:
        ws = S.get('wheelSpinVel', [0, 0, 0, 0])
        if ((ws[2] + ws[3]) - (ws[0] + ws[1])) > 2:
            accel -= 0.1
    return max(0.0, accel)

def destringify(s):
    if not s: return s
    if type(s) is str:
        try:    return float(s)
        except ValueError: return s
    elif type(s) is list:
        if len(s) < 2: return destringify(s[0])
        else:          return [destringify(i) for i in s]

# ==============================================================================
# 4. COMUNICAZIONE TRAMITE UDP
# ==============================================================================
class ServerState:
    def __init__(self):
        self.servstr = str()
        self.d = dict()

    def parse_server_str(self, server_string):
        self.servstr = server_string.strip()[:-1]
        sslisted = self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = destringify(w[1:])

class DriverAction:
    def __init__(self):
        self.actionstr = str()
        self.d = {
            'accel': 0.2, 'brake': 0, 'clutch': 0,
            'gear': 1, 'steer': 0,
            'focus': [-90, -45, 0, 45, 90], 'meta': 0,
        }

    def clip_to_limits(self):
        self.d['steer']  = clip(self.d['steer'],  -1, 1)
        self.d['brake']  = clip(self.d['brake'],   0, 1)
        self.d['accel']  = clip(self.d['accel'],   0, 1)
        self.d['clutch'] = clip(self.d['clutch'],  0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]: self.d['gear'] = 0
        if self.d['meta'] not in [0, 1]: self.d['meta'] = 0
        if (type(self.d['focus']) is not list or min(self.d['focus']) < -180 or max(self.d['focus']) > 180):
            self.d['focus'] = 0

    def __repr__(self):
        self.clip_to_limits()
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not type(v) is list: out += '%.3f' % v
            else: out += ' '.join([str(x) for x in v])
            out += ')'
        return out

class Client:
    def __init__(self, p=3001):
        self.host     = 'localhost'
        self.port     = p
        self.sid      = 'SCR'
        self.maxSteps = 100000
        self.S = ServerState()
        self.R = DriverAction()
        self.setup_connection()

    def setup_connection(self):
        self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.so.settimeout(1)
        while True:
            a = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = '%s(init %s)' % (self.sid, a)
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
                if '**identified**' in sockdata:
                    print(f"Connesso al server TORCS sulla porta {self.port}!")
                    break
            except socket.error:
                print(f"In attesa del server TORCS sulla porta {self.port}...")
                time.sleep(1)

    def get_servers_input(self):
        if not self.so: return
        while True:
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
                if '**identified**' in sockdata: continue
                elif not sockdata: continue
                else:
                    self.S.parse_server_str(sockdata)
                    break
            except socket.error:
                pass

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error:
            pass

    def shutdown(self):
        if not self.so: return
        self.so.close()
        self.so = None


# ==============================================================================
# 5. MAIN: ALGORITMO KNN E LAYER EURISTICO
# ==============================================================================
if __name__ == "__main__":
    print("Inizializzazione e addestramento del cervello KNN di base...")
    knn_driver = KNNDriver(k=29)
    
    
    lista_dataset = glob.glob("data/*.csv")
    
    if not lista_dataset:
        print("\n[ERRORE] Nessun file .csv trovato nella cartella corrente!")
        print("Assicurati di avere i dati estratti per far funzionare il KNN.")
        sys.exit(1)

    print(f"Trovati {len(lista_dataset)} file dataset. Inizio l'addestramento veloce...")
    knn_driver.train_model(lista_dataset)

    
    C = Client(p=3001)
    
    
    R_state = {'steer': 0.0, 'accel': 0.2, 'brake': 0.0, 'clutch': 0.0, 'gear': 1, 'meta': 0}

    print("\nMacchina pronta sulla griglia. Inizio del giro autonomo!")
    try:
        for step in range(C.maxSteps, 0, -1): 
            C.get_servers_input()
            S, R = C.S.d, C.R.d
            
            if not S: 
                continue

            
            rule_steer = calculate_steering(S)
            rule_accel = calculate_throttle(S, R_state)
            rule_brake = apply_brakes(S)

            
            speed_x = S.get('speedX', 0)
            track_pos_raw = S.get('trackPos', 0)
            angle_raw = S.get('angle', 0)

            knn_preds = knn_driver.predict(speed_x, track_pos_raw, angle_raw)
            knn_steer = knn_preds[0] * 1.5
            knn_accel = knn_preds[1]
            knn_brake = knn_preds[2]

           
            base_steer = (rule_steer * 0.55) + (knn_steer * 0.45)
            base_accel = (rule_accel * 0.55) + (knn_accel * 0.45)
            
            base_accel = traction_control(S, base_accel)
            base_brake = max(rule_brake, knn_brake) 

            
            final_steer = float(np.clip(base_steer, -1.0, 1.0))
            final_accel = float(np.clip(base_accel, 0.0, 1.0))
            final_brake = float(np.clip(base_brake, 0.0, 1.0))

            
            R_state['steer'] = final_steer
            R_state['accel'] = final_accel
            R_state['brake'] = final_brake

            
            R['steer'] = final_steer
            R['accel'] = final_accel
            R['brake'] = final_brake
            R['gear']  = shift_gears(S)

            C.respond_to_server()

    except KeyboardInterrupt:
        print("\nGiro di test interrotto dall'utente.")
    finally:
        C.shutdown()
        print("Connessione interrotta correttamente.")