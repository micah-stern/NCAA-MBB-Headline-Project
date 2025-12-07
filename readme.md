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

### Key Features
- Scrapes box score data from ESPN's public APIs
- Generates natural language headlines from numerical statistics
- Two model implementations: custom transformer and fine-tuned BART
- Evaluation using BLEU-4 and ROUGE-L metrics

---

## Models

### Model 1: Custom Transformer (From Scratch)
**Architecture:** Custom encoder-decoder transformer built with TensorFlow/Keras
- **Input:** 50 × 12 matrix (top 50 players, 12 features per player)
- **Features:** team_id, minutes, points, rebounds, assists, turnovers, steals, blocks, fouls, winner_flag
- **Training Data:** ~700 games (Nov 3-17, 2025)
- **Training:** 20 epochs, categorical cross-entropy loss, Adam optimizer

**Results:**
| Metric | Score |
|--------|-------|
| BLEU-4 | 6.18 × 10⁻⁷⁹ |
| ROUGE-L | 0.185 |
| Token F1 | 0.032 |
| Exact Match | 0.0% |

**Key Issues:**
- Insufficient training data for learning English from scratch
- Hallucinated team names and scores
- Overfit to high-frequency tokens ("no.", "bench", etc.)

### Model 2: Fine-tuned BART (facebook/bart-base)
**Architecture:** Pre-trained BART-base (140M parameters) fine-tuned on NCAA data
- **Why BART?** Pre-trained on summarization tasks, better than T5 (already headline-trained)
- **Training:** 3 epochs with checkpoints
- **Input Format:** Text prompts describing game context and key player stats

**Results:**
| Metric | Score |
|--------|-------|
| BLEU-4 | >0.15 |
| ROUGE-L | >0.60 |

**Sample Headlines:**
- ✅ "Murray scores 24 points, No. 19 Ole Miss beats Long Island 80-64"
- ✅ "Kohler's 20 lead Michigan State past Niagara 80-64"
- ✅ "Pryce Sandfort scores 22 points as No. 12 Iowa beats Southern 80-64"

**Observations:**
- Checkpoint 3 showed signs of overfitting (repetitive diction)
- Required specific prompt engineering to reduce hallucinations
- Generated ESPN-quality headlines with proper basketball terminology

---

## Installation

### Prerequisites
- Python 3.8+
- GPU recommended (but not required)

### Setup
```bash
# Clone the repository
git clone https://github.com/micah-stern/NCAA-MBB-Headline-Project.git
cd NCAA-MBB-Headline-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries
```txt
tensorflow>=2.10.0
torch>=2.0.0
transformers>=4.30.0
pandas
numpy
nltk
scikit-learn
requests  # For ESPN API scraping
```

---

## Usage

### 1. Data Scraping
Collect box scores and headlines from ESPN:
```bash
python data/scraping/scrape_espn.py --start_date 2025-11-03 --end_date 2025-11-17
```

This generates:
- `data/boxscores.csv` - Player statistics
- `data/headlines.csv` - ESPN headlines

### 2. Train Custom Transformer
```bash
cd models/custom_transformer
python train.py --data ../../data --epochs 20 --batch_size 32
```

### 3. Fine-tune BART Model
```bash
cd models/bart_finetuned
python train.py --model facebook/bart-base --data ../../data --epochs 3
```

### 4. Generate Headlines
```bash
cd models/bart_finetuned
python generate.py --checkpoint checkpoint-3 --game_id 401706926
```

**Interactive Mode:**
```bash
python generate.py --interactive
```
Then enter prompts like:
```
Generate a sports headline for the game between Duke Blue Devils and Army Black Knights. 
Key performers: Kon Knueppel (Duke): 15 pts, 2 reb, 2 ast
```

---

## Data Format

### Box Score Input (per game)
| Player | Team | MIN | PTS | FG | 3PT | FT | REB | AST | TO | STL | BLK | PF |
|--------|------|-----|-----|----|----|----|----|-----|----|----|-----|-----|
| Cameron Boozer | Duke | 29 | 35 | 13-16 | 2-2 | 7-11 | 12 | 5 | 1 | 3 | 3 | 1 |

### Target Headline
```
"Boozer has 35 points and 12 rebounds as No. 4 Duke beats Indiana State 100-62"
```

---

## Evaluation Metrics

### BLEU-4
Measures n-gram overlap between generated and reference headlines
- **Range:** 0 to 1 (higher is better)
- **BART Model:** >0.15 (good for sports headlines with infinite vocabulary)

### ROUGE-L
Measures longest common subsequence
- **Range:** 0 to 1 (higher is better)
- **BART Model:** >0.60 (excellent phrase-level overlap)

---

## Key Findings

### What Worked
✅ BART pre-training dramatically improved results (6.18×10⁻⁷⁹ → 0.15+ BLEU)  
✅ Model learned basketball terminology ("double-double", "lead", "past")  
✅ Correctly identified AP Poll rankings (e.g., "No. 4", "No. 12")  
✅ Generated grammatically correct, ESPN-style headlines  
✅ Captured game narratives and key player performances

### Challenges
❌ Custom transformer: insufficient data (~700 games) to learn English from scratch  
❌ Hallucinated team names when teams not in training distribution  
❌ Checkpoint 3 overfitting reduced headline diversity  
❌ Required careful prompt engineering to avoid hallucinations  
❌ Difficulty handling proper nouns (player/team names)

### Future Improvements
1. **More training data:** Expand to full season (~5,800 games)
2. **Custom tokenization:** Single tokens for team names (e.g., "North_Carolina_State")
3. **Additional features:** Team rankings, win streaks, rivalry context
4. **Play-by-play integration:** Add game flow narrative beyond box scores
5. **Hybrid approach:** Combine structured data extraction with LLM generation

---

## Examples

### Custom Transformer (Poor Results)
```
Input:  Game 401826071 - South Carolina 87, Radford 58
Output: "no. scores scores for and in the bench in ot carolina 2nd-half win over new"
Issues: Nonsensical, wrong teams, hallucinated "ot"
```

### BART Model (Strong Results)
```
Input:  Generate headline for Ole Miss vs Long Island. Murray: 24 pts, 3 ast
Output: "Murray scores 24 points, No. 19 Ole Miss beats Long Island 80-64"
Issues: None - accurate, ESPN-style headline
```

---

## Citation

If you use this project, please cite:
```
Uwazurike, U., & Stern, M. (2025). NCAA Men's Basketball Headline Generation 
using Transformer Models. ECE684 Final Project.
```

---

## Team

- **Uzoma Uwazurike** - Model development, evaluation
- **Micah Stern** - Data scraping, training pipeline

**Course:** ECE684 - Natural Language Processing  
**Institution:** Duke University  
**Semester:** Fall 2025

---

## License

This project is for educational purposes. ESPN data used under fair use for academic research.

---

## Acknowledgments

- ESPN for public API access
- Hugging Face for pre-trained BART model
- Course instructor for feedback on improving the model

---

## Links

- **GitHub Repository:** https://github.com/micah-stern/NCAA-MBB-Headline-Project
- **BART Model:** https://huggingface.co/facebook/bart-base
- **T5 Reference:** https://huggingface.co/JulesBelveze/t5-small-headline-generator

