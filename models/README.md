# Model dosyaları

Dashboard ve değerlendirme komutlarının doğrudan çalışabilmesi için iki küçük
dağıtım modeli Git'e dahil edilir:

- `tactical_dqn_goal_aware.pth`: Tactical DQN Scout politikası
- `happo_scratch_final.pth`: tamamlanmış HAPPO Scout/Hunter politikası

`happo_*_episode_*.pth`, `happo_best.pth` ve diğer ara checkpoint dosyaları
yerel eğitim çıktılarıdır; `.gitignore` içinde kalırlar. HAPPO değerlendirmesi,
mevcut tamamlanmış modeller arasından en son yazılanı otomatik seçer.

Checkpoint dosyalarını yeniden üretmek için proje kökünde:

```bash
./.venv/bin/python train_tactical_dqn.py
./.venv/bin/python train_happo.py --mode scratch
```
