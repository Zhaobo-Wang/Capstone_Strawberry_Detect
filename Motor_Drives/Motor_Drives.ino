const int motorPin = LED_BUILTIN;    // LED pin number

void setup() {
  // initialize the LED pin as an output:
  pinMode(motorPin, OUTPUT);
  // initialize serial communication:
  Serial.begin(115200);
}

void loop() {
  
  if (Serial.available() > 0) {
    
    char incomingByte = Serial.read();
    // read the incoming byte:
    // 打印出传入的字节
    Serial.print("Python Value: ");
    Serial.println(incomingByte);
    
    // check if the incoming byte is 'H':
    if (incomingByte == 'H') {
      // turn LED on:
      digitalWrite(motorPin, LOW);
    } else if (incomingByte == 'L') {
      // turn LED off:
      digitalWrite(motorPin, HIGH);
    }
  }
}
