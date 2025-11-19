# Headline Generation for NCAA Men’s Basketball Statistics

    Statistics play a critical role in sports, providing valuable insights into player performance and
game outcomes. However, box scores and raw numerical data can be difficult to interpret,
making it challenging for fans and analysts to quickly grasp the story of the game. Automating
the process of turning game-wise statistics into summaries and headlines can increase the
accessibility and appeal of the ever-changing landscape of sports.

In particular, NCAA Basketball boasts a season-wide, Division 1 game total of 5,800. This
number can be daunting, and thus, we will use Natural language processing to input a curated
dataset of previous headlines and box scores, to produce novel headlines for games occurring in
the 2025-26 season. This data will be found legally through ESPN’s APIs, creating a bivariate
dataset with real input stats and target headlines. The data can be encoded into numerical,
categorical, or textual features as necessary. Utilizing an LSTM Encoder-Decoder (Seq2Seq),
categorical cross-entropy loss, and beam search, we can generate output sequences word by
word quickly and efficiently. Lastly, our model will be evaluated using human judgement, and
factual accuracy tests.
