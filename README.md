1. Context și obiectiv

Tema lucrării: „Interfețele de utilizator în robotica chirurgicală, incluzând comunicația industrială între echipamente”.

Obiectiv: realizarea unei interfețe gestuale care să controleze, prin gesturi simple ale mâinii, un sistem (simulat) de robot / echipament, cu:

procesare video locală pe Raspberry Pi 5,

accelerare AI pe modulul Hailo-8 (26 TOPS),

transmisie de comenzi prin TCP/IP către un PC/server.

2. Arhitectura hardware

Raspberry Pi 5 (8 GB RAM) – platformă edge.

Modul Hailo-8 conectat la RPi (PCIe) – rulare rețea neuronală hand-landmark.

Cameră USB pentru captură imagine.

PC / laptop pe post de server TCP, care primește comenzile și loghează / vizualizează rezultatele.

Rețea locală (LAN / Wi-Fi), cu server la IP ex: 192.168.88.252, port 5005.

3. Stack software

Sistem de operare: Raspberry Pi OS.

Limbaj: Python 3.11.

Biblioteci principale:

opencv-python – captură video, prelucrare imagine, UI.

mediapipe – detecția landmark-urilor mâinii.

hailo_platform – API HailoRT (versiunea 4.20) pentru rularea modelului .hef.

numpy, pandas, matplotlib – procesare date și analiză.

Logger personalizat (logger.py) care scrie în CSV:

timestamp, fps, frame_time_ms

hailo_score, hailo_valid

mediapipe_landmarks

command, tcp_sent, tcp_reconnected

cx, cy (centroidul conturului în ROI)

4. Procesing pipeline (logică pe RPi)
4.1 Captură și ROI

Camera capturează continuu cadre de la VideoCapture(0).

Se definește o zonă de interes (ROI) fixă în imagine: ROI_X1, ROI_Y1, ROI_X2, ROI_Y2.

ROI este zona unde utilizatorul poziționează mâna pentru a genera comenzi.

4.2 Segmentarea clasică a pielii

Conversie ROI la spațiul HSV.

Aplicare prag:

lower = np.array([0, 20, 70])
upper = np.array([20, 255, 255])
mask = cv2.inRange(hsv, lower, upper)


Operații morfologice (dilate + GaussianBlur).

Căutare contur maxim și calcul centroid (cx, cy).

4.3 Mapare centroid → comenzi

ROI se împarte în 3 zone verticale:

stânga → Move Left

centru → Centered

dreapta → Move Right

Dacă nu există contur suficient de mare → No hand.

4.4 Filtru AI (scenariul cu Hailo)

ROI este redimensionat la 224×224 și convertit la RGB.

ROI este inferat pe modulul Hailo prin modelul hand_landmark_lite.hef, folosind:

configurare via ConfigureParams.create_from_hef(...)

InputVStreams / OutputVStreams cu format UINT8.

Din output-ul modelului se calculează un hailo_score (media valorilor normalizate).

Dacă hailo_score < prag (HAILO_SCORE_THRESHOLD) → sistemul consideră că nu are o mână validă și forțează comanda No hand.

Altfel, comanda generată de logica clasică este acceptată și transmisă.

4.5 MediaPipe (landmark-uri)

În paralel, sistemul rulează MediaPipe Hands pe ROI (sau pe tot frame-ul).

Se obține mediapipe_landmarks ca număr de puncte detectate (de regulă 21).

Punctele sunt desenate în UI pentru vizualizare (puncte verzi în fereastra video).

4.6 Transmisia comenzilor (TCP)

Un mic client TCP (CommandSender) se conectează la PC-ul server.

La schimbarea comenzii (command != last_cmd), sistemul trimite string-ul (cu \n) către server.

Sunt logate:

tcp_sent (1 dacă s-a trimis o comandă în acel frame)

tcp_reconnected (1 dacă a fost nevoie de reconectare).

4.7 Limită de sesiune

Pentru analize reproducibile, fiecare script (AI și NO-AI) trimite maxim 100 de comenzi, după care se oprește automat.

5. Scripturi principale

rpi_hand_ui_hailo_ai.py:

Pipeline complet: cameră → ROI → segmentare → AI (Hailo) → MediaPipe → comandă → TCP → logging CSV.

Hailo rulează la fiecare frame, scorul este logat în hailo_score.

rpi_hand_ui_hailo_noai.py:

Aceeași logică vizuală (segmentare + MediaPipe), dar fără Hailo.

Valori hailo_score = 0, hailo_valid = 0 în CSV → permite comparația directă.

6. Analiza datelor (notebook)

Notebook-ul încarcă session_ai.csv și session_no_ai.csv.

Se construiește un timp relativ t_sec bazat pe timestamp.

Se calculează și afișează:

FPS mediu, minim, maxim.

Latență medie și maximă pe cadru.

Distribuția comenzilor (Move Left, Move Right, Centered, No hand).

Procentul de cadre No hand (robustețe).

Distribuția centroidului în ROI (stabilitatea gesturilor).

Scorul mediu Hailo și raportul de cadre hailo_valid.

Se generează un tabel sumar comparativ (AI vs NO-AI) și se exportă în .csv și .md.

7. Concluzii (template)

După ce rulezi notebook-ul și vezi valorile reale, poți formula concluzii de genul (aici tu vei înlocui cu cifrele tale):

În scenariul fără AI (NO-AI), sistemul a obținut un FPS mediu de ~X fps, cu o latență medie de ~Y ms per cadru.
În scenariul cu AI (Hailo ON), FPS-ul mediu a fost de ~A fps, iar latența medie a crescut/ scăzut la ~B ms, datorită introducerii etapei suplimentare de inferență pe acceleratorul Hailo-8.

Filtrul AI a redus procentul de cadre în care gestul a fost interpretat incorect sau în care fundalul era confundat cu mâna, fapt evidențiat de scăderea / creșterea procentului de cadre etichetate „No hand” de la P_noai la P_ai.
Scorul mediu Hailo (hailo_score) de ~S indică faptul că modelul răspunde diferit în funcție de prezența sau absența mâinii în ROI, ceea ce confirmă funcționarea corectă a pipeline-ului AI.

Per ansamblu, integrarea modulului Hailo-8 a permis realizarea unei interfețe gestuale filtrate de un model AI edge, menținând în același timp timpi de răspuns compatibili cu utilizarea într-o interfață în timp aproape real, specifică roboților chirurgicali și echipamentelor industriale.
