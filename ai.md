这是starVLA的代码仓库，我正在把le-wm的世界模型嵌入这个框架，我的主要目标是提升libero数据集的成功率。
你需要尝试各种可能的结果来提升libero评测中的实验成功率。基线的路径在 playground/Checkpoints/lewm_oft_libero_wm_oft_future_aw05_state0_vitft_10k/checkpoints/steps_8000_pytorch_model.pt, 基线的成功率约为 83 % (浮动6%) , starVLA里用Qwen做VLM，OFT head的成功率约 96 , 路径在  playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt 你也可以蒸馏这个模型
你可以在基线的基础上每次训练1个或多个epoch，在训练之后去跑评测。如果训练还没有结束，就等待训练完成。训练可以用8卡，OOM的话就用0-3。
如果你准备修改模型尺寸，需要从头训练，也可以多用一些时间，从头训练一般用80000个step. 如果是在已有的模型上微调，则跑30000个step。已有的实验表明微调的效果并不显著。
评测以前看一下训练是否有效，你改的部分loss是否有明显下降。
评测数据一般用libero goal的100个episode, 统计最终成功率
如果成功率比基线高，就把代码推送到 my-origin 的 dev.ai 分支上（没有就创建一个），否则就 git reset 到初始代码。
实验过程不需要停下来问我是否继续之类的问题，我可能在睡觉，你应该完全自主迭代。
你需要把每次的idea和实验过程、结果写在experiment.md文件里。你的idea应该要宽一些，你可以修改模型结构, loss设置等过程，不要只局限于改超参和系数。
如果你没有优化的idea了，就去读最近的世界模型的论文，从不同角度来优化现有的模型、框架和参数。尝试从不同角度来解决问题。