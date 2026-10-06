# Historical training targets and provenance

These are target observations to compare against a fresh run. They are not a single verified, internally consistent run record yet.

| Observation | Reported value | Evidence status |
|---|---:|---|
| Demo corpus | 20 theorem files; 408,359 characters/tokens; vocabulary 133 | Transcript output embedded in `archive/Hannah.py` |
| Model size | 3,062,016 trainable parameters; 4 layers, 4 heads, 256 dimensions | Transcript output / source configuration |
| Batches per epoch | 25,515 at batch size 16 | Transcript output; consistent with 408,231 next-character windows |
| Early loss | 4.86 to 3.06 by batch 106 | Historical analysis text in the archive; run remained in progress there |
| Epoch 1 duration | 14:42:55 for 25,515/25,515 | Reported in the October 4 reconstruction summary; raw epoch log/checkpoint not recovered in this archive |
| Later progress | Epoch 2 at 7,303/25,515; latest loss 2.4427; minimum 1.3113 | Reported in the October 4 reconstruction summary; raw batch log/checkpoint not recovered in this archive |
| Other corpus description | 118 theorem files; 1.66M characters | Separate embedded summary; unresolved whether this describes another corpus/run |

Do not collapse the 20-file and 118-file descriptions. A current run should record its corpus file hashes, vocabulary mapping, software version, seed, model parameter count, per-batch loss, epoch means, elapsed times, and checkpoint hashes. Reproduction of the reported loss values is not demonstrated until those source conditions are recovered and a new run completes.
