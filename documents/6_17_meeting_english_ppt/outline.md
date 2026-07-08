# Distinguishing Smile Types with Deep Learning

Draft outline for approval. No slide images or PPTX have been generated yet.

## Slide 1: Research Update

- Key points:
  - Distinguishing polite, bitter, and true smiles from facial video data
  - June 17 meeting update
  - Focus: dataset preparation, baseline modeling, and model design options
- Visual idea: Clean title slide with a subtle facial-analysis motif and three smile-category labels.
- Layout role and intent: Cover; frame the topic and meeting purpose.
- Required source images: None.

## Slide 2: Starting Point from Last Week

- Key points:
  - Current task: train a model using the existing dataset to classify smile types
  - Target classes: polite smile, bitter smile, and true smile
  - The long-term goal is to identify separate smile axes in the learned representation
  - Ideally, a pure smile type should increase linearly along its own axis
- Visual idea: Three horizontal representation axes, one for each smile type.
- Layout role and intent: Context; restate the prior conclusion and research direction.
- Required source images: None.

## Slide 3: Role of the DNN

- Key points:
  - Smile changes in raw video are affected by nonlinear noise
  - Noise sources include head motion, occlusion, and interaction-related movement
  - The DNN should reduce these nonlinear factors
  - The expected output is a more linear and separable smile representation
- Visual idea: Pipeline from noisy facial sequences to cleaner smile representation axes.
- Layout role and intent: Concept explanation; clarify why deep learning is needed.
- Required source images: None.

## Slide 4: This Week's Work

- Key points:
  - Organized and cleaned the current dataset
  - Split the data and performed data augmentation
  - Reviewed related papers
  - Compared possible model structures for the next baseline
- Visual idea: Weekly progress board with four completed work blocks.
- Layout role and intent: Progress update; show concrete work completed this week.
- Required source images: None.

## Slide 5: Current Dataset Status

- Key points:
  - Polite smile: 200 samples
  - Bitter smile: 100 samples
  - True smile: 50 samples
  - The true-smile subset is still too small for a reliable baseline
- Visual idea: Simple three-bar comparison chart emphasizing class imbalance.
- Layout role and intent: Data evidence; make the current data distribution explicit.
- Required source images: None.

## Slide 6: Dataset Limitations

- Key points:
  - The dataset is imbalanced across smile categories
  - Excessive movement changes head pose and facial angle
  - Interaction actions can remove or obscure important facial details
  - These factors may weaken classification and representation learning
- Visual idea: Three risk cards: imbalance, pose variation, and facial-detail loss.
- Layout role and intent: Risk analysis; explain what limits the current baseline.
- Required source images: None.

## Slide 7: Ideal Dataset Conditions

- Key points:
  - For each participant, collect balanced samples across categories
  - Target distribution: 100 bitter smiles, 100 polite smiles, and 100 true smiles
  - Keep the face mostly frontal by constraining facial angle
  - Avoid actions that occlude the main facial organs
- Visual idea: Ideal data checklist beside a balanced 3-class distribution diagram.
- Layout role and intent: Requirements; define the dataset target for stronger modeling.
- Required source images: None.

## Slide 8: Candidate Model Designs

- Key points:
  - Option 1: VGG-Face feature extractor followed by LSTM or TCN classification
  - Option 2: model the face as a graph and use graph convolution to propagate information across facial regions
  - Then use LSTM or TCN to model temporal changes
  - After completing the baseline dataset, test multiple DNN methods and compare smile-type differences
- Visual idea: Side-by-side architecture comparison ending in a next-step arrow.
- Layout role and intent: Architecture and next steps; close with the modeling plan.
- Required source images: None.
