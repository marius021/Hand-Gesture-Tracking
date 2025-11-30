Sistemul dezvoltat integrează mai multe componente software și hardware, care colaborează pentru a realiza o interfață gestuală în timp real, accelerată hardware și capabilă să comunice cu un echipament distant (robot industrial sau simulator). Arhitectura finală cuprinde următoarele elemente:
Raspberry Pi 5 – platforma principală de procesare (edge device).


Acceleratorul AI Hailo-8 conectat prin PCIe, utilizat ca filtru AI pentru validarea prezenței mâinii.


Camera video (USB) care furnizează fluxul de imagini.


Pipeline de viziune clasică (OpenCV) pentru segmentarea pielii și determinarea centrului de masă al mâinii.


Modelul MediaPipe Hands (rulat pe CPU) pentru obținerea landmark-urilor 3D ale mâinii (21 de puncte pe nodurile degetelor).


Modulul de comunicație industrială TCP/IP, care transmite comenzile discrete către un „robot” / PC.


Modulul de logging, care stochează în format CSV parametrii relevanți pentru analiză ulterioară.
