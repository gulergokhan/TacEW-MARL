# TacEW-MARL

TacEW-MARL, elektronik harp destekli bir eskort görevini GridWorld üzerinde
inceleyen araştırma prototipidir. Scout platformu radarları bastırır veya
aldatır; Hunter ise hedefe, Scout'un eskort menzilini ve radar riskini gözeterek
ilerler. Projede tek ajanlı Tactical DQN ile CTDE tabanlı HAPPO karşılaştırılır.

## Kurulum

Python 3.12 ile proje kökünde:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Komutlar proje kökünden çalıştırılmalıdır. `PYTHONPATH` sorunu yaşamamak için
aşağıdaki biçim kullanılabilir:

```bash
PYTHONPATH="$PWD" ./.venv/bin/python -m unittest discover -s test -v
```

## Dashboard

```bash
PYTHONPATH="$PWD" ./.venv/bin/python dashboard_server.py
```

Ardından tarayıcıda
`http://127.0.0.1:8000/tacew_dashboard.html` adresini açın. Dashboard üzerinden:

- DQN veya HAPPO eğitimi başlatılabilir ve durdurulabilir.
- Holdout değerlendirmeleri çalıştırılabilir.
- Scenario Builder ile 10x10–40x40 özel harita hazırlanabilir.
- DQN, HAPPO veya sezgisel politika ile özel sortie üretilebilir.
- Üretilen episode'lar Sortie Playback içinde adım adım incelenebilir.

HAPPO eğitimi 1. episode'u, her çıktı aralığındaki episode'u ve geri yüklenen
en iyi son politikayı playback kayıtlarına ekler. Özel HAPPO kayıtlarında yalnız
en son koşu tutulur. Özel senaryolarda DQN ve HAPPO; eskort kopması, gereksiz
jamming ve ölümcül radar rotalarını engelleyen taktik yürütme korumasıyla
birlikte çalışır. Bu durum episode etiketinde `tactical guard` olarak açıkça
gösterilir.

## Temel komutlar

```bash
# Tactical DQN
PYTHONPATH="$PWD" ./.venv/bin/python train_tactical_dqn.py
PYTHONPATH="$PWD" ./.venv/bin/python evaluate_tactical_dqn.py
PYTHONPATH="$PWD" ./.venv/bin/python evaluate_tactical_holdout.py

# HAPPO (ayrı scratch checkpoint'leri)
PYTHONPATH="$PWD" ./.venv/bin/python train_happo.py --mode scratch
PYTHONPATH="$PWD" ./.venv/bin/python evaluate_happo.py
PYTHONPATH="$PWD" ./.venv/bin/python evaluate_happo_holdout.py

# Bilimsel karşılaştırma: yürütme koruması olmadan yalnız ham HAPPO aktörleri
PYTHONPATH="$PWD" ./.venv/bin/python evaluate_happo_holdout.py --raw

# Mevcut en iyi HAPPO modelini onarma/devam ettirme
PYTHONPATH="$PWD" ./.venv/bin/python train_happo.py --mode resume

# Dashboard'dan indirilen özel senaryo
PYTHONPATH="$PWD" ./.venv/bin/python run_scenario.py scenario.json --policy dqn
PYTHONPATH="$PWD" ./.venv/bin/python run_scenario.py scenario.json --policy happo
```

## Test ve doğrulama

```bash
PYTHONPATH="$PWD" ./.venv/bin/python -m unittest discover -s test -v
PYTHONPATH="$PWD" ./.venv/bin/python -m pip check
```

Test paketi DQN, replay buffer, HAPPO güncellemesi, eskort korumaları, özel
senaryo doğrulaması, radar izleri, jamming, arazi ve dashboard episode depolama
davranışlarını kapsar. Holdout komutları eğitimde kullanılmayan başlangıç,
hedef ve radar yerleşimlerinde rota ezberini ayrıca ölçer. HAPPO holdout'un
varsayılan modu üretimde kullanılan taktik korumalı yürütmedir; `--raw` yalnız
aktör ağlarının genellemesini ölçer ve bu iki sonuç ayrı raporlanmalıdır.

## Dizinler

- `agent/`: DQN ve replay buffer
- `marl/`: HAPPO, aktör/critic ağları, rollout buffer ve yürütme korumaları
- `environment/`: GridWorld ve TacticalEnv görev mantığı
- `radar/`, `terrain/`, `weather/`, `aircraft/`: simülasyon bileşenleri
- `test/`: otomatik testler
- `models/`: dağıtım modelleri ve yerel eğitim checkpoint'leri
- `dashboard_logs/`: playback ve özel senaryo kayıtları

Bu proje bir araştırma/simülasyon prototipidir; gerçek uçuş, radar veya operasyon
karar sistemi değildir.
