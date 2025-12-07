# Zihao_Liu_CE301_University_of_Essex
My project is about using machine learning methods to design better SCMA codebooks. Among which some better large-scale MED calculation approaches is mentioned.

10/15/2025-Upload a code(paper_Train.py). It is about how to reproduce the results of ReinforcementLearning-Based SCMA Codebook Design for Uplink Rayleigh Fading Channels by Yen-Ming Chen.

10/27/2025-upload a draft for some MED prediction methods. It seems promising approaches.

11/15/2025-After some experiments. Those results turned out to be unsuitable for random Guass Codebooks.
           (Might be useful for those codebooks who are in good structure) Will try it when I am available.
           
           But for scale of 4 by 6 codebook, Pytorch has good optimization for calculation a martrix of 4096*4096. So for my main project, using exact GRAM martrix or Enumeration should be enough.
           
11/23/2025-Upload a file named reinforcement_learning_for_MED_prediction.py. It mainly use Supervised learning which has A regression net and a policy net. To predict how to sample the MED when calculation the pairs. The time complexity which is linear is promising  and The error rate of MED is about 10% to 12%. But a large scale training might be an exhausting work.
           Will try it when I am available.
           
12/06/2025-Upload a file named machine_learning_superposed_codebook.py. That is a early access of my work, in which I implemented gradient desent. It seems to increase the MED from Geometry aspect but actually the performance in AWGN channel simulation is almost the same as the one that is not optimized.
           Needs to find more policies to train.
