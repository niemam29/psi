# Ocena pojazdów na podstawie zdjęć za pomocą sieci neuronowych

## Wymagania

Przed rozpoczęciem pracy z projektem upewnij się, że masz zainstalowane:
- **Node.js** (zalecana wersja: 16+)
- **NPM** (w zestawie z Node.js)

## Instalacja

1. Sklonuj repozytorium (na upelu brakuje danych treningowych ze względu na rozmiar):
   ```bash
   git clone https://github.com/niemam29/psi.git
   cd nazwa-repo
   ```
2. Zainstaluj zależności:
   ```bash
   npm install
   ```

## Uruchamianie modelu

### Trening modelu
Uruchom jedno z poniższych poleceń, aby wytrenować model:
- **Model A (CNN)**:
  ```bash
  npm run trainA
  ```
- **Model B (MobileNet + Dense)**:
  ```bash
  npm run trainB
  ```

### Przewidywanie wyniku dla pojedynczego obrazu
- **Model A (CNN)**:
  ```bash
  npm run predictA
  ```
- **Model B (MobileNet + Dense)**:
  ```bash
  npm run predictB
  ```
  *(Domyślnie używa pliku `test2.png`, można zmienić plik wejściowy w skrypcie)*

### Ewaluacja modeli
Aby porównać wyniki modeli, uruchom:
```bash
npm run evaluate
```

