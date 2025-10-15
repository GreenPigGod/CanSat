
#include <Arduino.h>
// ==== Pins (PH/EN x 2) ====
const int PH1_PIN = 32;   // Motor1 PHASE
const int EN1_PIN = 33;   // Motor1 ENABLE (PWM)
const int PH2_PIN = 25;   // Motor2 PHASE
const int EN2_PIN = 26;   // Motor2 ENABLE (PWM)

// ==== PWM ====
const int PWM_FREQ = 20000;  // 20 kHz
const int PWM_RES  = 8;      // 0..255

// ==== Serial ====
String rxLine;

// ==== Prototypes ====
void setSpeed(int m, int v);       // m=1|2, v=-255..+255
void setSpeedBoth(int v);          // v=-255..+255
void setSpeedPercent(int m, int p);// p=-100..+100
void coast(int m);                 // EN=0（惰性停止）
void coastBoth();
void brake(int m);                 // ソフトブレーキ（素早く0へランプ）
void brakeBoth();

// ==== Setup ====
void setup() {
  Serial.begin(115200);

  pinMode(PH1_PIN, OUTPUT);
  pinMode(PH2_PIN, OUTPUT);

  // 新APIスタイル：pinごとに周波数/分解能まで一発指定
  ledcAttach(EN1_PIN, PWM_FREQ, PWM_RES);
  ledcAttach(EN2_PIN, PWM_FREQ, PWM_RES);

  coastBoth();

  Serial.println("=== Serial Dual Motor Control (PH/EN x2) ===");
  Serial.println("数値: -255..255 -> 両モータ速度/方向");
  Serial.println("1NNN / 2NNN: それぞれのモータ速度 (例: 1127, 2-200)");
  Serial.println("pNN: -100..100% -> 両モータ (例: p50)");
  Serial.println("p1NN / p2NN: それぞれ%指定 (例: p1-80)");
  Serial.println("NNN,MMM: 両モータ同時 (例: -120,200)");
  Serial.println("b: brake(両) / b1 / b2");
  Serial.println("c: coast(両) / c1 / c2");
}

// ==== Loop ====
void loop() {
  // 1行受信（LF で確定）
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      rxLine.trim();
      if (rxLine.length() > 0) {
        // 解析
        String s = rxLine;
        rxLine = "";

        // 1) "NNN,MMM" 形式（両モータ同時）
        int comma = s.indexOf(',');
        if (comma >= 0) {
          int v1 = s.substring(0, comma).toInt();
          int v2 = s.substring(comma + 1).toInt();
          setSpeed(1, constrain(v1, -255, 255));
          setSpeed(2, constrain(v2, -255, 255));
          Serial.printf("[OK] M1=%d, M2=%d\r\n", v1, v2);
          return;
        }

        // 2) 単文字コマンド群
        if (s.equalsIgnoreCase("b")) { brakeBoth(); Serial.println("[OK] brake both"); return; }
        if (s.equalsIgnoreCase("c")) { coastBoth(); Serial.println("[OK] coast both"); return; }
        if (s.equalsIgnoreCase("b1")){ brake(1);   Serial.println("[OK] brake M1");   return; }
        if (s.equalsIgnoreCase("b2")){ brake(2);   Serial.println("[OK] brake M2");   return; }
        if (s.equalsIgnoreCase("c1")){ coast(1);   Serial.println("[OK] coast M1");   return; }
        if (s.equalsIgnoreCase("c2")){ coast(2);   Serial.println("[OK] coast M2");   return; }

        // 3) pNN / p1NN / p2NN（%指定）
        if (s.length() >= 2 && (s[0] == 'p' || s[0] == 'P')) {
          if (s[1] == '1' || s[1] == '2') {
            int m = s[1] - '0';               // 1 or 2
            int pct = s.substring(2).toInt(); // -100..100
            setSpeedPercent(m, pct);
            Serial.printf("[OK] M%d=%d%%\r\n", m, pct);
            return;
          } else {
            int pct = s.substring(1).toInt(); // -100..100
            setSpeedPercent(1, pct);
            setSpeedPercent(2, pct);
            Serial.printf("[OK] both=%d%%\r\n", pct);
            return;
          }
        }

        // 4) 1NNN / 2NNN（片側の速度）
        if (s.length() >= 2 && (s[0] == '1' || s[0] == '2')) {
          int m = s[0] - '0';             // 1 or 2
          int v = s.substring(1).toInt(); // -255..255
          setSpeed(m, constrain(v, -255, 255));
          Serial.printf("[OK] M%d=%d\r\n", m, v);
          return;
        }

        // 5) 単数値（両モータ同じ速度）
        {
          bool numeric = true;
          for (size_t i = 0; i < s.length(); ++i) {
            char c = s[i];
            if (!(c == '-' || isdigit(c))) { numeric = false; break; }
          }
          if (numeric) {
            int v = s.toInt();
            setSpeedBoth(constrain(v, -255, 255));
            Serial.printf("[OK] both=%d\r\n", v);
            return;
          }
        }

        // 不明コマンド
        Serial.println("[ERR] unknown cmd");
      }
    } else {
      rxLine += ch;
    }
  }
}

// ==== Helpers ====

static inline void _applyPHEN(int phPin, int enPin, int v) {
  int mag = abs(v);
  digitalWrite(phPin, (v >= 0) ? HIGH : LOW);
  ledcWrite(enPin, mag); // 0..255
}

void setSpeed(int m, int v) {
  if (m == 1) _applyPHEN(PH1_PIN, EN1_PIN, v);
  else        _applyPHEN(PH2_PIN, EN2_PIN, v);
}

void setSpeedBoth(int v) {
  v = constrain(v, -255, 255);
  _applyPHEN(PH1_PIN, EN1_PIN, v);
  _applyPHEN(PH2_PIN, EN2_PIN, v);
}

void setSpeedPercent(int m, int p) {
  p = constrain(p, -100, 100);
  int v = map(p, -100, 100, -255, 255);
  setSpeed(m, v);
}

void coast(int m) {
  if (m == 1) { ledcWrite(EN1_PIN, 0); } // EN=0 -> Hi-Z（惰性）
  else         { ledcWrite(EN2_PIN, 0); }
}

void coastBoth() { coast(1); coast(2); }

// PH/ENモードでは本当の“短絡ブレーキ”ができないため、素早く0に落とすソフトブレーキ
void brake(int m) {
  // 現在の向きは問わず、短いランプダウンで0へ
  // 実機に合わせてステップやディレイは調整してください
  for (int a = 200; a >= 0; a -= 25) {
    setSpeed(m, (digitalRead((m==1)?PH1_PIN:PH2_PIN)==HIGH) ? a : -a);
    delay(2);
  }
  coast(m);
}
void brakeBoth() { brake(1); brake(2); }

