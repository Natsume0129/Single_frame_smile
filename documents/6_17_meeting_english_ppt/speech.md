## Slide 1: Research Update

Today I want to give a short update on the smile-type classification project. The main research goal is to distinguish polite smiles, bitter smiles, and true smiles from facial video data. This week, the focus has been on preparing the dataset, thinking through the first baseline, and narrowing down the model structures I want to test next.

---

Presenter cues:
- Emphasize that this is a research progress update, not a final result presentation.
- Point first to the three smile categories, then to the dataset and baseline focus.

## Slide 2: Starting Point from Last Week

The starting point from last week is that the task is not only to classify smile types. The broader goal is to learn a representation where different smile types can be separated along different axes. Ideally, if a smile is a pure example of one category, its change should become more linear along the corresponding axis.

This is why I am thinking about both classification accuracy and the structure of the learned feature space. The model should help us understand how different smiles vary, not only assign a label.

---

Presenter cues:
- Slow down on the idea of "separate smile axes."
- Explain that linearity is a desired property of the learned representation.

## Slide 3: Role of the DNN

The raw video signal contains many sources of nonlinear variation. A smile can change together with head motion, occlusion, and interaction-related movement, so the facial sequence is not a clean measurement of smile intensity or smile type.

Here, the role of the DNN is to suppress variation that is not directly related to the smile. If it works well, the output representation should be cleaner, more separable, and closer to the linear smile axes we discussed on the previous slide.

---

Presenter cues:
- Guide the audience from left to right: raw sequence, DNN, cleaner representation.
- Stress that the DNN is used as a representation cleaner as well as a classifier.

## Slide 4: This Week's Work

This week I worked on the preparation needed before training the baseline model. I organized and cleaned the current dataset, split the data, and performed augmentation. In parallel, I reviewed related papers and compared model structures that could be used for the next experiments.

The important point is that the work is now moving from data preparation into baseline training and model comparison.

---

Presenter cues:
- Keep this slide brief.
- Use it as a bridge from preparation work to dataset status.

## Slide 5: Current Dataset Status

The current dataset has 200 polite-smile samples, 100 bitter-smile samples, and 50 true-smile samples. This means the dataset is usable for initial experiments, but the class balance is still a major constraint.

The true-smile subset is the weakest part at the moment. I originally expected that around 100 true-smile samples would be enough to build a first baseline, but with only 50 samples, the baseline may be less stable.

---

Presenter cues:
- Point to the three bars and make the imbalance explicit.
- Emphasize that true-smile data is the current bottleneck.

## Slide 6: Dataset Limitations

There are three main dataset limitations I am considering. First, class imbalance weakens the reliability of the baseline. Second, excessive movement means the head is not always facing forward, which introduces extra pose variation. Third, actions during interaction can obscure or remove important facial details.

All of these factors can reduce classification quality and also make the learned representation less clean. So even if the model trains, we need to interpret baseline results with these limitations in mind.

---

Presenter cues:
- Present the three cards as separate sources of noise.
- Link the limitations back to representation quality, not only accuracy.

## Slide 7: Ideal Dataset Conditions

For a stronger dataset, I would ideally want balanced samples for each participant: 100 bitter smiles, 100 polite smiles, and 100 true smiles. I would also want the face to remain mostly frontal and avoid actions that occlude the main facial organs.

This slide describes the target condition rather than the current condition. It gives a clearer standard for what kind of data would make the baseline and later representation analysis more reliable.

---

Presenter cues:
- Make clear that this is an ideal target, not the current dataset.
- Mention that these conditions are especially important for participant-level comparison.

## Slide 8: Candidate Model Designs

For the model, I am currently considering two directions. The first is to continue using VGG-Face as the feature extractor, followed by LSTM or TCN to model temporal changes. This is the more direct baseline path.

The second direction is to model the face as a graph, use graph convolution to propagate information across facial regions, and then use LSTM or TCN for temporal modeling. After the baseline dataset is completed, I plan to test these DNN methods and compare how well they separate the different smile types.

---

Presenter cues:
- Compare Option 1 and Option 2 as baseline versus graph-based extension.
- End with the next action: train DNN baselines and compare smile-type differences.
