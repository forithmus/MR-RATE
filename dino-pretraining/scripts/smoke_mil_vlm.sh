#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MRRATE_VLM_DEVICE=cpu TOKENIZERS_PARALLELISM=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(dirname "$SCRIPT_DIR")"
: "${DINOV3_ROOT:?Set DINOV3_ROOT to the pinned official DINOv3 checkout}"
export PYTHONPATH="$CODE:$DINOV3_ROOT:${PYTHONPATH:-}"
R=${1:-$(mktemp -d /tmp/mrrate-vlm-smoke.XXXXXX)}
python3 "$CODE/tests/make_vlm_lifecycle_fixture.py" "$R"
python3 -m torch.distributed.run --standalone --nproc-per-node=2 -m mr_dino.vlm normalize --config "$R/config.json" --output "$R/normalization.pt"
python3 -m torch.distributed.run --standalone --nproc-per-node=2 -m mr_dino.vlm train --config "$R/config.json" --output "$R/uninterrupted"
python3 -m torch.distributed.run --standalone --nproc-per-node=2 -m mr_dino.vlm train --config "$R/config.json" --output "$R/resumed" --max-steps 1
python3 -m torch.distributed.run --standalone --nproc-per-node=2 -m mr_dino.vlm train --config "$R/config.json" --output "$R/resumed"
python3 - "$R" <<'PY'
import sys,torch,json
from pathlib import Path
r=Path(sys.argv[1]);a=torch.load(r/'uninterrupted/latest.pt',weights_only=False);b=torch.load(r/'resumed/latest.pt',weights_only=False)
for group in ['adapter','visual','text']:
 for key in a[group]:torch.testing.assert_close(a[group][key],b[group][key],rtol=0,atol=0)
for key in ['temperature','pool_temperature']:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
assert a['step']==b['step']==3 and a['epoch']==b['epoch']==1
result={'synthetic_implementation_test':True,'world':2,'training_studies':5,'validation_studies':2,'steps':3,'includes_padded_tail':True,'normalization':'train only','checkpoint_resume':'bitwise identical final adapter, projector, BERT and temperatures'}
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
PY
