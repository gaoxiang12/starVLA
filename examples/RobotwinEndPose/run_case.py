"""Use the existing fixed-scene scoring loop with the feedback-only adapter."""
import sys
from examples.RobotwinEndPose import interface
from examples.Robotwin.audits.run_qwen_gawm_ranking_case import main

if __name__ == '__main__':
    # The shared runner imports this name inside main; isolated to this process.
    sys.modules['model2robotwin_interface'] = interface
    main()
