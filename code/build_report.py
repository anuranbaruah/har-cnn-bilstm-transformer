"""Build the Final Report (Part C) skeleton from the interim docx cover page + real results."""
import copy, json, os, sys
import numpy as np
from docx import Document
from docx.shared import Pt, RGBColor, Cm, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = "ICT_4442_Interim_Report_HAR_DRAFT.docx"
RES = sys.argv[1]
OUT = sys.argv[2]
S = json.load(open(os.path.join(RES, "summary.json")))
M = ["cnn", "bilstm", "transformer"]
N = {"cnn": "1D-CNN", "bilstm": "BiLSTM + attention", "transformer": "Transformer encoder"}
OWN = {"cnn": "Anuran Baruah", "bilstm": "Syed Ayaan Shere", "transformer": "Raghav Nagpal"}
CL = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
NOTE = RGBColor(0x80, 0x80, 0x80)
FILL = RGBColor(0xC0, 0x00, 0x00)

doc = Document(SRC)
body = doc.element.body
paras = doc.paragraphs
# keep cover (paragraphs 0..35), drop everything after
cut = paras[36]._element
started = False
for el in list(body):
    if el is cut: started = True
    if started and el.tag != qn("w:sectPr"):
        body.remove(el)
for p in doc.paragraphs:
    for r in p.runs:
        if r.text.strip() == "Interim report on" or "Interim report on" in r.text:
            r.text = r.text.replace("Interim report on", "Final report on")
for s in doc.sections:
    s.page_width, s.page_height = Mm(210), Mm(297)

st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def para(text="", size=10, bold=False, italic=False, align=None, color=None, smallcaps=False, font=None, space_after=4):
    p = doc.add_paragraph()
    if text:
        add_runs(p, text, size, bold, italic, color, smallcaps, font)
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    return p


def add_runs(p, text, size=10, bold=False, italic=False, color=None, smallcaps=False, font=None):
    # [[...]] marks a red FILL IN placeholder
    parts = text.split("[[")
    for i, part in enumerate(parts):
        if i == 0:
            seg = [(part, False)]
        else:
            a, _, b = part.partition("]]")
            seg = [("[" + a + "]", True), (b, False)]
        for t, is_fill in seg:
            if not t: continue
            r = p.add_run(t); r.font.size = Pt(size); r.bold = bold or is_fill; r.italic = italic
            r.font.small_caps = smallcaps
            if font: r.font.name = font
            if is_fill: r.font.color.rgb = FILL
            elif color: r.font.color.rgb = color


def h1(text):
    p = para(text, 10, align=WD_ALIGN_PARAGRAPH.CENTER, smallcaps=True, space_after=4)
    p.paragraph_format.space_before = Pt(10)
    return p


def h2(text):
    p = para(text, 10, italic=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    p.paragraph_format.space_before = Pt(6)
    return p


def note(text):
    """Grey italic writing note: replace with the team's own prose."""
    p = para("Note: " + text, 9, italic=True, color=NOTE, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=2)
    return p


def bullets(items, size=9.5):
    for it in items:
        p = para("", size, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=1)
        p.paragraph_format.left_indent = Cm(0.5); p.paragraph_format.first_line_indent = Cm(-0.3)
        add_runs(p, "• " + it, size, color=NOTE if False else None)


def grid(t):
    tblPr = t._element.tblPr; b = OxmlElement("w:tblBorders")
    for e in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        x = OxmlElement(f"w:{e}"); x.set(qn("w:val"), "single"); x.set(qn("w:sz"), "4"); x.set(qn("w:color"), "808080"); b.append(x)
    tblPr.append(b)


def shade(cell, hexcolor):
    tcPr = cell._element.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcolor); tcPr.append(sh)


def table(caption, header, rows, widths=None, size=8):
    para(caption, 8, smallcaps=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    t = doc.add_table(rows=1, cols=len(header)); grid(t); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; add_runs(c.paragraphs[0], h, size, bold=True); shade(c, "E8EEF7")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; add_runs(cells[i].paragraphs[0], str(v), size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths): row.cells[i].width = Cm(w)
    para("", 4, space_after=2)
    return t


def figure(path, caption, width_cm=17):
    p = para("", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=1); p.add_run().add_picture(path, width=Cm(width_cm))
    para(caption, 8, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)


def pm(m, k, pct=False, d=2):
    mu, sd = S[m][k]
    return f"{mu*100:.{d}f} ± {sd*100:.{d}f}" if pct else f"{mu:.{d}f} ± {sd:.{d}f}"



def P(text):
    return para(text)


c_, b_, t_ = (S[m] for m in M)

# ---------------- title block ----------------
doc.add_page_break()
para("Deep Learning-Based Human Activity Recognition from Smartphone Inertial Sensors: "
     "A Comparative Study of 1D-CNN, BiLSTM, and Transformer Architectures", 24, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
para("Anuran Baruah (220911544), Syed Ayaan Shere (230911015), Raghav Nagpal (230911196)", 11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=1)
para("School of Computer Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education, Manipal, India",
     10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=1)
para("[[FILL IN: email addresses]]", 9, font="Courier New", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)

# ---------------- Abstract ----------------
p = para("", 9); add_runs(p, "Abstract: ", 9, bold=True, italic=True)
add_runs(p, "Many deep learning models have been proposed for human activity recognition (HAR) from smartphone sensors, but they are usually evaluated on "
 "different datasets, splits and metrics, which makes it hard to tell which architecture family is best suited to the task. "
 "This project compares three families under one protocol on the UCI HAR dataset: a one-dimensional convolutional neural "
 "network (1D-CNN), a two-layer bidirectional LSTM with attention pooling (BiLSTM), and a Transformer encoder. All models "
 "take the raw 128 x 9 inertial windows as input, share the same subject-wise split (nine unseen test subjects), the same "
 "preprocessing and the same training budget, and are each trained with three random seeds. The 1D-CNN achieved the best "
 f"test macro F1 of {pm('cnn','macro_f1',True)}%, ahead of the BiLSTM ({pm('bilstm','macro_f1',True)}%) and the Transformer "
 f"({pm('transformer','macro_f1',True)}%). It was also the fastest model, at {c_['latency_ms_b1'][0]:.2f} ms per window on a "
 f"single CPU thread, compared with {b_['latency_ms_b1'][0]:.2f} ms for the BiLSTM and {t_['latency_ms_b1'][0]:.2f} ms for the "
 "Transformer. The three models differ mainly on the dynamic activities, where the CNN's local filters separate walking, "
 "upstairs and downstairs almost perfectly. All three share the same main failure: confusion between sitting and standing, "
 "which accounts for 53% to 83% of their errors, and errors overall are concentrated in two of the nine test subjects. The results suggest "
 "that the local inductive bias of convolution suits short, multichannel inertial windows better than recurrence or global "
 "self-attention when training data is limited.", 9, bold=True)
p = para("", 9); add_runs(p, "Keywords: ", 9, bold=True, italic=True)
add_runs(p, "human activity recognition, inertial sensors, 1D convolutional neural network, bidirectional LSTM, "
            "Transformer encoder, self-attention, UCI HAR, time-series classification", 9, bold=True)

# ---------------- I. Introduction ----------------
h1("I. Introduction")
P("Human activity recognition (HAR) is the task of identifying what a person is doing, such as walking, sitting or climbing "
  "stairs, from the signals of body-worn or embedded sensors. Smartphones are a natural platform for HAR because almost "
  "every phone contains an accelerometer and a gyroscope. Reliable HAR supports a wide range of applications, including "
  "fitness and health tracking, monitoring of elderly people living alone, fall detection, rehabilitation, and phones that "
  "adapt their behaviour to the user's context.")
P("Classical HAR pipelines compute a large set of hand-designed statistical and frequency features from each window of "
  "sensor data and feed them to a shallow classifier. The UCI HAR dataset, for example, was released with 561 such "
  "features and a support vector machine baseline [1]. Designing these features requires domain knowledge and may not "
  "transfer well to new users or devices. Deep learning models remove this step by learning features directly from the "
  "raw signals, and convolutional, recurrent and attention-based networks have all been reported to perform well on "
  "sensor-based HAR [2], [5], [6], [9], [14].")
P("The difficulty is that these results are hard to compare. Published models are evaluated on different datasets, with "
  "different train and test splits, different metrics and different amounts of tuning. A large study by Hammerla et al. [5] "
  "found that no single model family wins on every dataset. As a result, the literature does not clearly answer which "
  "architectural bias, local convolution, sequential recurrence or global self-attention, is best suited to short "
  "multichannel inertial signals.")
P("This project addresses that question with a controlled comparison. We implement one model from each family, a 1D-CNN, "
  "a BiLSTM with attention pooling, and a Transformer encoder, and evaluate all three on the UCI HAR dataset under an "
  "identical protocol: the same raw input, the same subject-wise split, the same preprocessing, the same training "
  "configuration and the same metrics, with three random seeds per model. Besides accuracy, we compare model size and "
  "inference latency and analyse the errors each model makes. The scope is limited to one dataset with six activities, "
  "a waist-mounted phone and offline evaluation on a CPU. Section II reviews related work, Section III states the "
  "objectives, Section IV describes the three models, Section V the experimental setup, Section VI the results and "
  "discussion, and Sections VII and VIII give conclusions and future work.")

# ---------------- II. Literature Review ----------------
h1("II. Literature Review")
h2("A. Dataset and classical baseline")
P("Anguita et al. [1] introduced the UCI HAR dataset, recorded from 30 volunteers carrying a smartphone at the waist, and "
  "reported 96% test accuracy using a multiclass support vector machine on 561 hand-crafted time and frequency features. "
  "Their per-class results already show the weakness that recurs throughout later work: recall for sitting was the lowest, "
  "at 88%, because sitting and standing produce very similar signals. This baseline is strong, so the main argument for "
  "deep models on this dataset is that they remove manual feature design rather than that they are far more accurate.")
h2("B. Convolutional models")
P("Yang et al. [3] were among the first to apply deep convolutional networks to multichannel sensor time series, "
  "convolving and pooling along the time axis, and showed that the learned features outperformed hand-crafted ones on the "
  "Opportunity and Hand Gesture datasets. Ronao and Cho [2] applied a deep 1D convolutional network to the raw UCI HAR "
  "signals and reached 94.79% test accuracy, rising to 95.75% when frequency-domain features were added. Ignatov [7] "
  "showed that a shallow CNN combined with a few statistical features can reach 97.63% on UCI HAR while remaining fast "
  "enough for real-time use. These works motivate our 1D-CNN and provide direct reference points on the same data.")
h2("C. Recurrent and hybrid models")
P("Recurrent networks model a window as an ordered sequence. The long short-term memory (LSTM) cell [11] addresses the "
  "vanishing gradient problem of plain recurrent networks, and bidirectional recurrent networks [12] read the sequence in "
  "both directions so that each step has access to past and future context. Attention pooling, introduced for machine "
  "translation by Bahdanau et al. [13], lets a model weight the most informative steps instead of relying on the last "
  "hidden state. Zhao et al. [6] used a deep residual bidirectional LSTM on UCI HAR and reported 93.57% accuracy. "
  "Ordóñez and Roggen [4] combined both ideas in DeepConvLSTM, four convolutional layers followed by two LSTM layers, and "
  "reported F1 scores of 0.915 on Opportunity and 0.958 on Skoda; this model was not evaluated on UCI HAR. Hammerla et "
  "al. [5] compared feed-forward, convolutional, LSTM and bidirectional LSTM models across thousands of runs on "
  "Opportunity, PAMAP2 and Daphnet Gait. The bidirectional LSTM was best on Opportunity, the CNN on PAMAP2 and the LSTM "
  "on Daphnet Gait, which shows that the best family depends on the data and on the protocol.")
h2("D. Attention models")
P("The Transformer [8] replaced recurrence with multi-head self-attention and positional encoding, allowing every position "
  "in a sequence to interact with every other position in a single layer. Mahmud et al. [9] adapted self-attention to "
  "wearable sensor data and reported macro F1 scores of 0.95 on PAMAP2 and 0.93 on Skoda, outperforming DeepConvLSTM on "
  "most of the datasets they tested. Dirgová Luptáková et al. [10] applied a Transformer encoder to smartphone motion "
  "data from the KU-HAR dataset and reported 99.2% average accuracy. Neither study used UCI HAR, so their numbers cannot "
  "be compared directly with ours.")
h2("E. Position of this work")
P("Two points follow from this review. First, reported accuracies on UCI HAR for single deep models lie between about 93% "
  "and 98% [2], [6], [7], close to the classical baseline [1], and are obtained under different protocols. Second, the "
  "strongest recent attention results [9], [10] come from other datasets. Surveys of deep learning for sensor-based HAR "
  "[14] note the same lack of common evaluation. Our contribution is therefore not a new architecture but a controlled "
  "comparison of the three families on one dataset, with a subject-wise split, a shared training budget, macro F1 as the "
  "main metric and results averaged over several seeds.")

# ---------------- III. Objectives ----------------
h1("III. Objectives")
for o in [
    "1. To compare a 1D-CNN, a BiLSTM and a Transformer encoder for six-class activity recognition on UCI HAR under a single reproducible protocol, using test macro F1 averaged over three seeds.",
    "2. To quantify the accuracy versus cost trade-off of each family through macro F1, per-class F1, parameter count and CPU inference latency.",
    "3. To identify the failure modes of each model, in terms of confused activity pairs and difficult subjects, and to explain the differences through the inductive bias of each architecture."]:
    P(o)

# ---------------- IV. Methodology ----------------
h1("IV. Methodology")
P("All three models take a normalised window of shape 128 timesteps by 9 channels as input and output scores for the six "
  "activities. They are implemented in PyTorch [15] and trained with the shared configuration in Section IV-D.")
arch = {
 "cnn": [("Input", "(batch, 128 timesteps, 9 channels), transposed to channels-first"),
         ("Blocks", "3 x [Conv1D kernel 5, padding 2, BatchNorm [17], ReLU, MaxPool 2]"),
         ("Channels", "64, 128, 128"), ("Pooling", "Global average pooling over time"),
         ("Head", "Dropout 0.3, Linear 128 -> 6"), ("Parameters", f"{c_['params']:,}")],
 "bilstm": [("Input", "(batch, 128 timesteps, 9 channels)"),
            ("Recurrent layers", "2 stacked bidirectional LSTM layers, hidden size 64 per direction, dropout 0.3 between layers"),
            ("Pooling", "Additive attention: Linear 128 -> 64, tanh, Linear 64 -> 1, softmax over time, weighted sum"),
            ("Head", "Dropout 0.3, Linear 128 -> 6"), ("Parameters", f"{b_['params']:,}")],
 "transformer": [("Input", "(batch, 128 timesteps, 9 channels)"),
                 ("Embedding", "Linear 9 -> 64 per timestep, plus fixed sinusoidal positional encoding"),
                 ("Encoder", "3 Transformer encoder layers, model width 64, 4 heads, feed-forward 128, dropout 0.1, post-norm"),
                 ("Pooling", "Mean over the 128 timesteps (no CLS token)"),
                 ("Head", "LayerNorm, Dropout 0.3, Linear 64 -> 6"), ("Parameters", f"{t_['params']:,}")],
}
text = {
 "cnn": [
  "The 1D-CNN treats the nine sensor channels as input channels and slides one-dimensional filters along the time axis. "
  "It has three convolutional blocks. Each block applies a convolution with kernel size 5 and same padding, batch "
  "normalisation [17], a ReLU activation and max pooling with a factor of 2. The number of filters grows from 64 to 128 "
  "and stays at 128 in the third block, so the temporal length shrinks from 128 to 16 while the number of feature maps "
  "grows. Global average pooling over the remaining 16 steps gives a 128-dimensional vector, which passes through dropout "
  "(0.3) and a linear layer to the six classes. The model has "
  f"{c_['params']:,} trainable parameters (Table I).",
  "A kernel of 5 samples covers 0.1 s at 50 Hz. After the three blocks, each output position depends on 36 input samples "
  "(about 0.72 s), which is close to the duration of one walking step. The model was chosen as the convolutional "
  "representative because its inductive bias, local and shift-invariant pattern detection, matches the short, repeated "
  "motion patterns that distinguish dynamic activities. Global average pooling makes the prediction independent of where "
  "in the window a pattern occurs and keeps the parameter count low. Convolutional models have performed well on the same "
  "dataset in prior work [2], [7]. The expected weakness is that the network has no explicit notion of order beyond its "
  "receptive field."],
 "bilstm": [
  "The BiLSTM reads the window as a sequence of 128 nine-dimensional vectors. Two stacked bidirectional LSTM layers [11], "
  "[12] with 64 hidden units per direction produce a 128-dimensional representation at every timestep, with dropout of 0.3 "
  "between the layers. Instead of keeping only the final hidden state, an additive attention layer [13] scores each "
  "timestep with a small network (a linear layer to 64 units, tanh, and a linear layer to one score), normalises the "
  "scores with a softmax over time, and forms a weighted sum of the hidden states. This summary vector passes through "
  "dropout (0.3) and a linear layer to the six classes. The model has "
  f"{b_['params']:,} trainable parameters (Table II).",
  "The BiLSTM was chosen as the recurrent representative because it models the order of the signal explicitly and, being "
  "bidirectional, gives every timestep access to both earlier and later context in the window. Attention pooling allows "
  "the model to focus on the most informative part of the window, which is useful when an activity is only clearly visible "
  "in part of it. A residual BiLSTM reported 93.57% on the same dataset [6], which gives a direct reference point. The "
  "expected cost is speed: the LSTM must process the 128 steps one after another, which limits parallelism at inference."],
 "transformer": [
  "The Transformer encoder first projects each nine-channel timestep to a 64-dimensional embedding with a linear layer and "
  "adds fixed sinusoidal positional encodings [8], so that the model knows the order of the timesteps. Three standard "
  "encoder layers follow, each with four-head self-attention, a feed-forward network of width 128, residual connections, "
  "layer normalisation and dropout of 0.1. The 128 output vectors are averaged over time, normalised, and passed through "
  "dropout (0.3) and a linear layer to the six classes. Mean pooling was used instead of a classification token to keep "
  f"the model simple. The model has {t_['params']:,} trainable parameters, the fewest of the three (Table III).",
  "The Transformer was chosen as the attention-based representative because self-attention lets every timestep interact "
  "with every other timestep in a single layer, without recurrence, so it can capture global structure in the window and "
  "can be computed in parallel. Attention-based models have reported strong HAR results on other datasets [9], [10]. Its "
  "inductive bias is weaker than that of convolution: it must learn which neighbouring timesteps matter from the data. "
  "With only 5,905 training windows, this was expected to be its main risk, and the model was kept small (width 64, three "
  "layers) for that reason."],
}
for i, m in enumerate(M):
    h2(f"{'ABC'[i]}. {N[m]} ({OWN[m]})")
    P(text[m][0])
    table(f"Table {['I','II','III'][i]}. {N[m]} configuration", ["Component", "Setting"], arch[m], widths=[3.5, 13])
    P(text[m][1])
h2("D. Common training configuration")
P("All three models are trained with exactly the same configuration (Table IV), so that differences in results can be "
  "attributed to the architectures. We use the Adam optimiser [16] with a learning rate of 0.001 and weight decay of "
  "0.0001, cross-entropy loss and a batch size of 64. Training runs for at most 40 epochs and stops early when the "
  "validation macro F1 has not improved for 8 epochs. The checkpoint with the best validation macro F1 is evaluated once "
  "on the test set. No hyperparameters were tuned on the test set.")
table("Table IV. Training configuration (identical for all models)", ["Setting", "Value"], [
    ("Optimiser", "Adam [16], learning rate 0.001, weight decay 0.0001"), ("Loss", "Cross-entropy, no class weights"),
    ("Batch size", "64, training set shuffled each epoch"), ("Epochs", "Up to 40, early stopping with patience 8 on validation macro F1"),
    ("Model selection", "Checkpoint with best validation macro F1, evaluated once on the test set"),
    ("Seeds", "42, 7, 123 (3 runs per model)"), ("Augmentation", "None"),
    ("Framework / hardware", "PyTorch [15]; CPU, 1 thread per run")], widths=[4, 12.5])

# ---------------- V. Experimental Setup ----------------
h1("V. Experimental Setup")
h2("A. Dataset")
para("All experiments use the UCI HAR dataset [1]. Thirty volunteers aged 19 to 48 carried a Samsung Galaxy S II "
     "on the waist while performing six activities: walking, walking upstairs, walking downstairs, sitting, standing "
     "and laying. The phone recorded 3-axis acceleration and 3-axis angular velocity at 50 Hz. The dataset authors "
     "filtered the signals and cut them into windows of 2.56 s (128 samples) with 50% overlap, giving 10,299 labelled "
     "windows. Each window has nine raw channels: body acceleration, gyroscope and total acceleration, each on the x, y "
     "and z axes. We use these raw signals as model input, not the 561 hand-crafted features released with the "
     "dataset, so that each network has to learn its own features.")
h2("B. Preprocessing and split")
para("The nine signal files are stacked into an array of shape (windows, 128, 9), and labels are mapped from 1-6 to "
     "0-5. The dataset has no missing values, so no windows were removed; we confirmed that the window counts match "
     "the official 7,352 training and 2,947 test windows. The split is done by subject (Table V). The official nine "
     "test subjects form the test set. From the 21 official training subjects, subjects 1, 11, 21 and 27 are held out "
     "for validation, which is used only for early stopping and checkpoint selection. No subject appears in more than "
     "one split, so test scores measure how a model performs on people it has never seen. Each channel is z-score "
     "normalised with the mean and standard deviation of the training split only, to avoid leaking information from "
     "the validation or test data. No data augmentation is used. The same loading, split and normalisation code "
     "(data.py) is shared by all three models.")
table("Table V. Data split (subject-wise, same for every model)", ["Split", "Subjects", "Windows"], [
    ("Train", "17 of the 21 official training subjects", "5,905"),
    ("Validation", "Subjects 1, 11, 21, 27", "1,447"),
    ("Test", "Official 9 test subjects (2, 4, 9, 10, 12, 13, 18, 20, 24)", "2,947")], widths=[2.5, 9.5, 2.5])
h2("C. Metrics")
para("We report test accuracy and macro-averaged precision, recall and F1. Macro F1 is the main metric because it "
     "gives the six classes equal weight. We also report per-class F1, confusion matrices and per-subject test "
     "accuracy for the error analysis. Cost is measured by the number of trainable parameters and by CPU inference "
     "latency: the median of 300 forward passes at batch size 1 (and 30 passes at batch size 256, divided by 256), "
     "on one CPU thread, timed one model at a time after all training had finished. Each model is trained with three "
     "random seeds (42, 7 and 123), and every score is given as the mean and sample standard deviation over the three "
     "runs.")

# ---------------- VI. Results ----------------
h1("VI. Results and Discussion")
h2("A. Comparison table")
para("Table VI compares the three models on the common metrics, Table VII lists each individual run, and Table VIII "
     "and Fig. 1 give the per-class F1 scores. Fig. 2 shows the confusion matrices for seed 42 and Fig. 3 the "
     "training curves for the same runs. The 1D-CNN has the highest mean test accuracy (93.15%) and macro F1 "
     "(93.32%) and the smallest spread across seeds, followed by the BiLSTM (90.71% and 90.90%) and the Transformer "
     "(88.01% and 87.93%). The 1D-CNN is also the fastest at batch size 1 (0.29 ms per window, against 1.70 ms for "
     "the BiLSTM and 1.28 ms for the Transformer). The Transformer has the fewest parameters (101,574) but the "
     "highest cost per window at batch size 256. All three models reach 99% or more macro F1 on the validation "
     "subjects, well above their test scores.")
rows = []
for m in M:
    rows.append((N[m], pm(m, "accuracy", True), pm(m, "macro_precision", True), pm(m, "macro_recall", True),
                 pm(m, "macro_f1", True), pm(m, "best_val_macro_f1", True), f"{S[m]['params']:,}",
                 f"{S[m]['latency_ms_b1'][0]:.2f}", f"{S[m]['latency_ms_per_window_b256'][0]:.3f}"))
table("Table VI. Test results, mean ± std over 3 seeds (all values in %, except parameters and latency)",
      ["Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "Val macro F1", "Params", "Latency b=1 (ms)", "Latency b=256 (ms/window)"],
      rows, size=7.5)
seedrows = []
for m in M:
    for r in sorted(S[m]["runs"], key=lambda r: r["seed"]):
        seedrows.append((N[m], r["seed"], f"{r['accuracy']*100:.2f}", f"{r['macro_f1']*100:.2f}", r["epochs_run"],
                         r["total_errors"], r["sit_stand_errors"]))
table("Table VII. Per-seed test results", ["Model", "Seed", "Accuracy (%)", "Macro F1 (%)", "Epochs run", "Test errors", "Sit/Stand errors"],
      seedrows, size=7.5)
pcrows = [(c, *[f"{S[m]['per_class_f1_mean'][c]:.3f}" for m in M]) for c in CL]
table("Table VIII. Per-class test F1, mean of 3 seeds", ["Activity", *[N[m] for m in M]], pcrows, size=8)
figure(os.path.join(RES, "fig_per_class_f1.png"), "Fig. 1. Per-class test F1 (mean of 3 seeds).", 15)
figure(os.path.join(RES, "fig_confusion_matrices.png"), "Fig. 2. Test confusion matrices, seed 42 (rows: true class, columns: predicted).")
figure(os.path.join(RES, "fig_training_curves.png"), "Fig. 3. Training loss and validation macro F1, seed 42.")
h2("B. Why performance differs")
P("The ranking is consistent across seeds. The gap between the 1D-CNN and the BiLSTM (about 2.4 points of macro F1) and "
  "between the BiLSTM and the Transformer (about 3.0 points) are both larger than the standard deviation of either model. "
  "With only three seeds per model we do not claim statistical significance, but the separation is clean: every CNN run "
  "(92.96% to 93.90% macro F1) scored above every BiLSTM run (90.20% to 92.18%), and every BiLSTM run scored above every "
  "Transformer run (86.33% to 89.63%) (Table VII).")
P("The differences come almost entirely from the dynamic activities. On sitting, standing and laying the three models "
  "perform similarly (Table VIII). On walking, however, the mean F1 is 0.994 for the CNN, 0.959 for the BiLSTM and 0.876 "
  "for the Transformer, and the number of confusions among walking, upstairs and downstairs per seed is about 15, 46 and "
  "155 respectively. Walking-type activities are defined by short, repeated motion patterns such as individual steps. A "
  "convolution with a 0.72 s receptive field is built to detect such local patterns wherever they occur in the window, "
  "so the CNN learns them quickly and reliably. The BiLSTM can represent them as well, but must do so through a "
  "sequential state that is updated one sample at a time, which is harder to optimise. The Transformer has no built-in "
  "notion of locality: it must learn from the data alone that neighbouring samples matter more than distant ones. With "
  "only 5,905 training windows, this weaker inductive bias appears to be a disadvantage, which agrees with the finding of "
  "Hammerla et al. [5] that the best family depends on the data rather than on model capacity.")
P("A second factor is the training protocol. All three models reach between 0.99 and 1.00 macro F1 on the four validation "
  "subjects but only 0.88 to 0.93 on the nine test subjects, so the validation subjects are much easier than the test "
  "subjects. Because the validation score saturates within a few epochs, early stopping selected very early checkpoints: "
  "the best epoch was between 2 and 7 in seven of the nine runs, and the Transformer runs stopped after 10 to 15 epochs. "
  "Validation scores therefore could not distinguish the models, and the Transformer in particular may be under-trained "
  "under the shared budget. We kept the protocol identical for all models rather than retuning one of them, since tuning "
  "the Transformer alone would make the comparison unfair.")
P("The cost results show a different trade-off. The CNN is the fastest model at batch size 1 (0.29 ms per window), about "
  "six times faster than the BiLSTM (1.70 ms), because its convolutions are computed in parallel over the whole window "
  "while the LSTM must process 128 steps in order. The Transformer has the fewest parameters (101,574) and is faster than "
  "the BiLSTM at batch size 1 (1.28 ms), but it is the most expensive per window when windows are batched (1.62 ms per "
  "window at batch size 256, against 0.66 ms for the BiLSTM and 0.16 ms for the CNN), because the cost of self-attention "
  "grows with the square of the sequence length. Parameter count alone is therefore a poor guide to inference cost.")
P("Compared with the literature, our CNN (93.15% mean accuracy) is below the 94.79% reported by Ronao and Cho [2], and our "
  "BiLSTM (90.71%) is below the 93.57% of Zhao et al. [6]. Our protocol is stricter in three ways: four training subjects "
  "are held out for validation, no hyperparameters are tuned, and we report the mean over three seeds rather than a single "
  "best run. Our numbers also differ slightly from the single runs in the interim report (CNN 92.98%, BiLSTM 91.14% for "
  "seed 42), most likely because the final runs used one CPU thread each, which changes the order of floating-point "
  "operations; re-running the interim configuration with the default thread count reproduced the interim CNN exactly. This "
  "sensitivity to small changes is one reason we report results over several seeds.")
h2("C. Error analysis")
figure(os.path.join(RES, "fig_per_subject_acc.png"), "Fig. 4. Test accuracy per unseen test subject (mean of 3 seeds).", 15)
P("Confusion between sitting and standing is the main failure mode of every model. Per seed, about 168 (CNN), 194 "
  "(BiLSTM) and 189 (Transformer) test windows are confused between these two classes, which is 83%, 71% and 53% of each "
  "model's errors respectively (Table VII, Fig. 2). Because all three families make roughly the same number of these "
  "errors, the limit appears to lie in the data rather than in any one architecture. Both postures produce almost "
  "motionless signals, and the main difference between them is the orientation of the phone relative to gravity, which "
  "changes little for a phone worn at the waist. The same weakness appears in the classical baseline, where sitting had "
  "the lowest recall [1].")
P("The remaining errors differ by model. The CNN and the Transformer almost never misclassify laying, while the BiLSTM "
  "misclassifies about 18 laying windows per seed, mostly as upstairs. The Transformer's second largest error source is "
  "walking predicted as downstairs (48 to 93 windows per seed) and, in seed 42, upstairs predicted as walking (58 windows). "
  "This matches the explanation above: the model has not learned the fine local differences between the three walking "
  "activities.")
P("Errors are also concentrated in a few people (Fig. 4). Subject 10 has a mean test accuracy of 74.7% with the CNN, "
  "65.3% with the BiLSTM and 61.7% with the Transformer, and subject 9 has 83.0%, 78.2% and 66.0%. In contrast, subjects 12, "
  "13, 18, 20 and 24 are recognised with 90% to 99% accuracy by all three models. Since the same two subjects are the "
  "hardest for every family, the cause is likely to be how these individuals perform the activities, for example how they "
  "sit or stand, rather than a defect in a particular model. This shows that a subject-wise split is essential, as a "
  "random split would hide this source of error.")

# ---------------- VII / VIII ----------------
h1("VII. Conclusion")
P("We compared a 1D-CNN, a BiLSTM with attention and a Transformer encoder for human activity recognition on the UCI HAR "
  "dataset under one protocol. The 1D-CNN was the best model on every measure we used: it had the highest test macro F1 "
  f"({pm('cnn','macro_f1',True)}%), the smallest variation across seeds and the lowest latency, with a moderate parameter "
  f"count. The BiLSTM was second ({pm('bilstm','macro_f1',True)}%) and the slowest at batch size 1, and the Transformer was "
  f"third ({pm('transformer','macro_f1',True)}%) despite having the fewest parameters.")
P("The differences arise on the dynamic activities, where the local, shift-invariant filters of the CNN capture step "
  "patterns that the recurrent and attention models learn less reliably from a small training set. All models share the "
  "same limitation, the confusion between sitting and standing, which appears to be a property of the data, and all "
  "struggle with the same two test subjects. For short, multichannel inertial windows and limited data, the local "
  "inductive bias of convolution is the best fit among the three families we tested.")
h1("VIII. Future Work")
P("This study has several limitations. It uses one dataset, three seeds per model, only four validation subjects, no "
  "hyperparameter search, and CPU latency only. Future work should first make validation more informative, for example "
  "with more validation subjects or leave-one-subject-out evaluation, so that early stopping and model selection are "
  "reliable, and then give each model a small, equal tuning budget. The sitting and standing confusion could be addressed "
  "with features that describe the orientation of gravity, longer windows or a class-weighted loss. Data augmentation such "
  "as jittering, scaling and rotation may improve robustness across subjects. Other architectures worth testing include a "
  "hybrid CNN-LSTM such as DeepConvLSTM [4] and a Transformer that operates on short patches of the signal, which adds "
  "local structure to attention. Finally, the models should be evaluated on a second dataset such as WISDM or KU-HAR and "
  "measured on a mobile GPU or phone, and compressed through quantisation or pruning for on-device use.")

h1("Acknowledgement")
P("We thank the faculty of the School of Computer Engineering, Manipal Institute of Technology, for their guidance during "
  "this mini project, and the creators of the UCI HAR dataset for making it publicly available.")
p = para(""); add_runs(p, "Use of AI tools: ", 10, bold=True)
add_runs(p, "An LLM (Claude, Anthropic) was used to write the shared code (data.py, models.py, train.py, eval_extra.py and the "
            "plotting scripts), to run the three-seed experiments, to produce the tables and figures, and to draft the text "
            "of this report. The team reviewed the code, results and text.")

h1("References")
refs = [
 "D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, “A public domain dataset for human activity recognition using smartphones,” in Proc. ESANN, Bruges, Belgium, 2013, pp. 437–442.",
 "C. A. Ronao and S.-B. Cho, “Human activity recognition with smartphone sensors using deep learning neural networks,” Expert Syst. Appl., vol. 59, pp. 235–244, 2016.",
 "J. B. Yang, M. N. Nguyen, P. P. San, X. L. Li, and S. Krishnaswamy, “Deep convolutional neural networks on multichannel time series for human activity recognition,” in Proc. IJCAI, 2015, pp. 3995–4001.",
 "F. J. Ordóñez and D. Roggen, “Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition,” Sensors, vol. 16, no. 1, art. no. 115, 2016.",
 "N. Y. Hammerla, S. Halloran, and T. Plötz, “Deep, convolutional, and recurrent models for human activity recognition using wearables,” in Proc. IJCAI, 2016, pp. 1533–1540.",
 "Y. Zhao, R. Yang, G. Chevalier, and M. Gong, “Deep residual Bidir-LSTM for human activity recognition using wearable sensors,” Math. Probl. Eng., vol. 2018, art. no. 7316954, 2018.",
 "A. Ignatov, “Real-time human activity recognition from accelerometer data using convolutional neural networks,” Appl. Soft Comput., vol. 62, pp. 915–922, 2018.",
 "A. Vaswani et al., “Attention is all you need,” in Proc. NeurIPS, 2017, pp. 5998–6008.",
 "S. Mahmud et al., “Human activity recognition from wearable sensor data using self-attention,” in Proc. ECAI, 2020, pp. 1332–1339.",
 "I. Dirgová Luptáková, M. Kubovčík, and J. Pospíchal, “Wearable sensor-based human activity recognition with transformer model,” Sensors, vol. 22, no. 5, art. no. 1911, 2022.",
 "S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, pp. 1735–1780, 1997.",
 "M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE Trans. Signal Process., vol. 45, no. 11, pp. 2673–2681, 1997.",
 "D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,” in Proc. ICLR, 2015.",
 "J. Wang, Y. Chen, S. Hao, X. Peng, and L. Hu, “Deep learning for sensor-based activity recognition: A survey,” Pattern Recognit. Lett., vol. 119, pp. 3–11, 2019.",
 "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Proc. NeurIPS, 2019, pp. 8024–8035.",
 "D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. ICLR, 2015.",
 "S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in Proc. ICML, 2015, pp. 448–456.",
]
for i, r in enumerate(refs, 1):
    p = para(f"[{i}] {r}", 8, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=1)
    p.paragraph_format.left_indent = Cm(0.6); p.paragraph_format.first_line_indent = Cm(-0.6)

doc.add_page_break()
h1("C.3: Individual Contribution Statement")
para("This page must be written by the members themselves and each member is individually vivaed on their stated contribution.", 9, italic=True)
table("Individual contribution statement", ["Member Name", "Reg. No.", "Model / Task Owned", "Key Contributions", "Signature"], [
    ("Anuran Baruah", "220911544", "1D-CNN", "[[FILL IN]]", ""),
    ("Syed Ayaan Shere", "230911015", "BiLSTM + attention", "[[FILL IN]]", ""),
    ("Raghav Nagpal", "230911196", "Transformer encoder", "[[FILL IN]]", "")], widths=[3, 2.2, 3, 6.5, 2.5], size=9)
p = para(""); add_runs(p, "GitHub repository: ", 10, bold=True); add_runs(p, "[[FILL IN: repo link]]")
doc.save(OUT)
print("saved", OUT)
