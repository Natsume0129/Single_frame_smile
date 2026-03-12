## 1
This week, my main work was to continue the method we discussed before. The basic idea is to define a true-smile axis, then see how other smiles move along this axis, and measure their projection and deviation.

Today, I will talk about my assumptions, my analysis method, the results, and my current conclusions.

## 2
First, these are my assumptions.

Different types of smiles have different temporal trajectory patterns in linear space.

Temporally, smiles can be seen as patterns that move from a non-smile region to a smile region in linear space.

True smile can provide a reference dynamic direction to measure how much other smiles advance along the main direction and how much they deviate from the true-smile path.

## 3
Based on these assumptions, we first extract raw features using VGG-Face fc7. Then after preprocessing and time normalization, we get a sequence.

Each sequence is made of 20 vectors, and each vector has 4096 dimensions.

Then I considered that directly comparing many sample pairs is not a good way, so it is necessary to find one typical case, or one prototype trajectory, for each class.

I mainly used two methods.

One method is to calculate a statistically typical curve by using the median.

The other method is to calculate distances between sequences and choose the real sequence with the smallest total distance, which means the sequence located in the most central position.

## 4
Here is the calculation method of Method A.

For each normalized time point, and for each feature dimension, I take the median value across all sequences. Then I combine the median values of all dimensions into one vector for that time point. I repeat this for all time points, and finally I get one prototype trajectory.

## 5
Then for Method B, as we discussed before, it is better if the example we use can correspond to a real case. So I also used this method.

I concatenate the 20 vectors of one sequence into one matrix.

For the distance between two sequences, I use the Frobenius norm.

The sequence with the smallest total distance to all other sequences is defined as the prototype trajectory example that we use for analysis.

## 6
Here are the three class prototypes obtained by the algorithm. They look quite different.

But because lighting and pose inside one sequence are roughly consistent, in preprocessing these shared factors are reduced, and what remains is more related to the change from the first frame.

## 7
Next is the definition of the smile main axis.

Simply speaking, vector g is the vector connecting the first point and the last point of the true-smile prototype. u is the unit vector that represents the direction.

As shown in the figure, the black curve represents the true-smile curve. We connect its beginning and end, and this gives us the main axis.

## 8
After defining the smile main axis, I mainly do three things.

First, I calculate the spatial distance at each time point. For two sequences, the norm of the difference vector between the two vectors at the same time point means how different they are in linear space. Here I use Euclidean distance.

In the figure, the black curve represents the true-smile prototype, and the red curve represents the polite-smile prototype. I calculate the absolute distance between them at each time point.

The other two parts are to calculate how much the other smile categories move along the true-smile axis, and how much they deviate from it.

## 9
As shown in the figure:

The x-axis is time. Because we resample into 20 points, the maximum value is 20.

The y-axis is distance. A larger distance means a larger overall expression difference between two points in space.

Here, the anchor means the reference sequence.

For example, in the top-left figure, I use the prototype obtained by Method A. At each time point, I calculate the distance from the polite-smile prototype and the ambiguous-smile prototype to the true-smile prototype.

For this result, I think the main points are:

The initial distances are usually small. This is more likely to reflect similarity in the neutral state, not smile differences themselves.

As time goes on, the distances between categories generally increase, which means dynamic differences become clearer after the smile unfolds.

Polite and ambiguous smiles are generally closer to each other, while both are clearly different from true smiles.

## 10
Next, I wanted to see how different smiles move along the smile axis.

For one time point, I first calculate the difference vector relative to the starting position.

Then I calculate its projection onto the smile main axis.

By calculating the ratio between the projection length and the length of the smile main axis g, I can see how far each time point has moved along the main axis.

## 11

## 12
The results of Method A and Method B are shown here.

The figures on the top only include the prototype trajectories.

The figures below also include the results of all real samples used in the calculation.

The dashed line represents the mean of the calculated results.

The light-colored region represents the middle 50 percent sample interval.

## 13
My conclusion here is as follows.

When using the median calculation in Method A, most true-smile data does move along the main axis.

In Method B, although this value is much smaller, we can still see numerical changes.

No matter whether we use Method A or Method B, the direction of progression of polite smile over time is clearly different from that of true smile.

By definition, ambiguous smile includes smiles that we cannot clearly classify, so it should be a transitional state between polite smile and true smile.

From the data, ambiguous smile does lie between true smile and polite smile, showing an intermediate state.

## 14
Finally, I calculated how much each sample deviates from the true-smile main axis.

The method is to construct the projection vector and then the difference vector between the original vector and the projection vector, and calculate the norm of this difference vector.

This ratio is the norm of the difference vector divided by the length of the true-smile main axis.

## 15
In the same way, the x-axis is time, and the y-axis is the ratio.

We can see that, starting from the prototype trajectories, the ratio usually becomes larger over time. This means the actual deviation becomes larger and larger.

The only exception is the last time point of true smile, and this comes directly from our definition.

I think the conclusion here can be:

The true smile still shows a significant deviation from the true-smile axis in the middle stages.

This means that the true-smile trajectory is not a straight line. The line connecting its beginning and end is only a rough reference direction.

Polite and ambiguous smiles progress less along the axis, but their off-axis deviation is considerable. This suggests that they are not static, but are mainly changing in other directions.

## 16
To sum up, I think the conclusion at this stage can be:

Different smile trajectories do differ in space.

Expressions in the neutral phase are more similar. As the smile becomes stronger, the similarity decreases.

Polite and ambiguous smiles move toward other regions in feature space rather than going to the true-smile region.

Polite and ambiguous smiles are closer to each other.

True smile shows a clear difference from the other two categories.
