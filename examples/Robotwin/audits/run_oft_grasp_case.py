"""Use the frozen physics/scoring runner with OFT's original input contract."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'examples/Robotwin/eval_files')]

from examples.Robotwin.audits import record_pregrasp_precision as recording
from examples.Robotwin.audits.oft_reference_adapter import CONTRACT, UNNORM_KEY, adapt_client
import model2robotwin_interface as interface


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        recording.case.main()
        return
    output = Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    original_get_model = interface.get_model

    def get_model(arguments):
        return adapt_client(original_get_model(dict(arguments, unnorm_key=UNNORM_KEY)))

    interface.get_model = get_model
    try:
        recording.case.main()
    finally:
        interface.get_model = original_get_model
        if len(recording.instances) == 1:
            recording.instances[0].save(output/'pregrasp_physics_poses.npz')
        if output.is_dir():
            recording.case.save(output/'input_contract.json', CONTRACT)
            for name in ('status.json', 'result.json'):
                path = output/name
                if path.is_file():
                    data = json.loads(path.read_text())
                    data['input_contract'] = CONTRACT
                    data['note'] = ('Dedicated red-block lift diagnostic, not full ranking. '
                                    'OFT receives the published natural-language task contract without state. '
                                    'Scene construction and passive physics scoring are unchanged.')
                    recording.case.save(path, data)


if __name__ == '__main__':
    main()
