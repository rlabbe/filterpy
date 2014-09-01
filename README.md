Kalman filters and other optimal and non-optimal estimation filters in Python.
--

This library provides Kalman filtering and various related optimal and
non-optimal filtering software written in Python. It contains Kalman
filters, Extended Kalman filters, Unscented Kalman filters, Kalman
smoothers, Least Squares filters, fading memory filters, g-h filters,
discrete Bayes, and more. 

This is code I am developing in conjunction with my book 
Kalman Filters and Random Signals in Python, which you can read/download at
http://rlabbe.github.io/Kalman-and-Bayesian-Filters-in-Python/

My aim is largely pedalogical - I opt for clear code that matches the
equations in the relevant texts on a 1-to-1 basis, even when that has a 
performance cost. There are places where this tradeoff is unclear - 
for example, I find it somewhat clearer to write a small set of equations 
using linear algebra, but numpy's overhead on small matrices makes it 
run slower than writing each equation out by hand, and books such as
Zarchan present the written out form, not the linear algebra form. It is 
hard for me to choose which presentation is 'clearer' - it depends on the
audience. In that case I usually opt for the faster implementation. 

I use numpy and scipy for all of the computations. I have experimented
with Numba, Continuum Analytics just in time compiler, and it yields 
impressive speed ups with minimal costs, but I am not convinced that I want
to add that requirement to my project. It is still on my list of things to
figure out, however.

As it evolves from alpha status I am adding documentation, tests, and examples,
but at the moment the my book linked above serves as the best documentation. I am
developing both in parallel, so one or the other has to suffer during the
development phase.  Reach out to me if you have questions or needs and I will
either answer directly or shift my development to address your problem (assuming
your question is a planned part of this library.


Basic use:
--
```
from filterpy.kalman import KalmanFilter
from filterpy.memory import FadingMemoryFilter


my_filter = KalmanFilter(3,4)
```

References
--

I use three main texts as my refererence, though I do own the majority of
the Kalman filtering literature. First is Paul Zarchan's 'Fundamentals of
Kalman Filtering: A Practical Approach'. I think it by far the best Kalman
filtering book out there if you are interested in practical applications
more than writing a thesis. The second book I use is Eli Brookner's 'Tracking
and Kalman Filtering Made Easy'. This is an astonishing good bood; its
first chapter is actually readable by the layperson! Brookner starts from the
g-h filter, and shows how all other filters - the Kalman filter, least squares,
fading memory, etc., all derive from the g-h filter. It greatly simplifies
many aspects of analysis and/or intuitive understanding of your problem. In
contrast, Zarchan starts from least squares, and then moves on to Kalman 
filtering. I find that he downplays the predict-update aspect of the algorithms,
but he has a wealth of worked examples and comparisons between different methods.
I think both viewpoints are needed, and so I can't imagine discarding one book.
Brookner also focuses on issues that are ignored in other books - track 
initialization, detecting and discarding noise, tracking multiple objects, an
so on.

I said three books. I also like and use Bar-Shalom's Estimation with Applications
to Tracking and Navigation. Much more mathmatical than the previous two books,
I would not recommend it as a first text unless you already have a background
in control theory or optimal estimation. Once you have that experience, this book
is a gem. Every sentence is crystal clear, his language is precise, but each
abstract mathematical statement is followed with something like "and this means...".


License
--
Copyright (c) 2014 Roger R Labbe Jr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
