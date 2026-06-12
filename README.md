#                                                     <h1 align="center">The-Pythonic-Five-Repository 🐍</h1>
<p align="center">
  <img width="500" height="273" alt="Logo Team Modificato" src="https://github.com/user-attachments/assets/97886542-236c-4cbe-859f-158ff4c6fbc2" />
</p>


📖 **Descrizione del Progetto**

Realizzazione di un agente di guida autononomo da parte del gruppo **The Pythonic Five**, in occasione della partecipazione alla competizione internazionale **IBM AI Racing League**. 

Il progetto è finalizzato all'implementazione di un sistema basato su intelligenza artificiale in grado di completare nel il tracciato **CorksCrew** del simulatore **TORCS** (**The Open Racing Car Simulator**).



🛠️ **Strumenti e Librerie Utilizzate:**

Per lo sviluppo dell'agente autonomo e la gestione dell'acquisizione dei dati, il progetto fa uso delle seguenti librerie Python:

* `scikit-learn`: Impiegata per implementare l'algoritmo **K-Nearest Neighbors (KNN) Regressor** e per la normalizzazione rapida dei dati in ingresso (tramite `StandardScaler`).
* `pandas`: Utilizzata per il pre-processing legato ai dati dei file CSV contenuti all'interno della cartella `data`.
* `numpy`: Impiegata nella gestione degli array multidimensionali.
* `pygame`: Adoperata nello script `manual_control_ds4.py` per stabilire l'interfaccia con l'hardware, catturando e mappando con altissima precisione gli input analogici del controller PS4 DualShock.
* `socket`: Inserita per stabilire la comunicazione UDP full-duplex stabile e a bassa latenza con il servet TORCS.
* **Librerie standard** (`os`, `glob`, `csv`): Utilizzate per il file handling.


🎮 **Raccolta dei dati:**

La raccolta del dataset è avvenuta ricorrendo allo script `manual_control_ds4.py`, che ha consentito di catturare i dati di sessanta giri di pista, effettuati ricorrendo ad un controller PS4 DualShock. Nello specifico, è stato effettuato il campionamento dei valori dei seguenti sensori:

* `speedX`
* `speedY`
* `speedZ`
* `angle`
* `gear`
* `rpm`
* `track`

I dati di ciascun lap sono stati salvati all'interno di un apposito file di log in formato Comma-Separated Values (CSV). L'insieme dei file CSV costituisce la base di addestramento dell'agente ed è contenuto all'interno della cartella `data`.






🧠 **Algoritmo impiegato: K-Nearest Neighbors Regressor**

L'addestramento e l'esecuzione dell'agente autonomo sono gestiti dallo script `myagent.py`, che fa uso dell'algoritmo di Machine Learning **K-Nearest Neighbors (KNN) Regressor**, implementato tramite la libreria `scikit-learn`.

In fase di avvio, lo script importa e concatena automaticamente tutti i file CSV presenti nella cartella `data`. Il modello viene quindi addestrato per stabilire una correlazione diretta tra la dinamica dell'auto sul tracciato e la sequenza ideale di comandi da eseguire.

Durante l'esecuzione, mentre si trova in pista, l'agente standardizza i dati dei sensori in tempo reale ed interroga il modello KNN. Quest'ultimo scansiona l'intero dataset per individuare le casistiche di guida passate più simili alla situazione corrente ed effettua l'inferenza dei valori ottimali di sterzata, accelerazione e frenata tramite l'interpolazione dei valori `target_steer`, `target_accel` e `target_brake`.





🧬 **Approccio Adottato: Behavioral Cloning**

L'approccio adottato per la realizzazione del progetto è il Behavioral Cloning, che consente di ridurre la complessità del problema di partenza tramite il ricorso ad un modello esperto da imitare, in un'ottica di apprendimento supervisionato. I vantaggi derivanti da tale approccio sono sostanziali:

* Riduzione del processo di sviluppo: la convergenza a un modello funzionante richiede tempi estremamente ridotti.

* Calo dei comportamenti imprevedibili: il modello non necessita di una corposa fase esplorativa per l'individuazione delle azioni e delle traiettorie funzionali.

* Facilità nell'integrazione di nuovi dati: se l'agente mostra incertezze o esce fuori pista in specifici punti del tracciato, è sufficiente ampliare l'insieme dei dati raccolti all'interno della cartella data mediante campionamenti ad hoc.




Questo setup ci consente di bilanciare la naturalezza delle traiettorie impostate dall'uomo con l'impeccabile precisione matematica della macchina in fase di frenata e controllo della trazione.
