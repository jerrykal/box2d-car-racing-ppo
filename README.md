# Box2D CarRacing PPO

Implementation of Proximal Policy Optimization(PPO) algorithm for the
[`CarRacing-v2`](https://gymnasium.farama.org/environments/box2d/car_racing/)
Box2D environment from OpenAI's Gym library.

## Demo

![demo](./gif/best_play.gif)

The demo is just one of the best play out of 1500 of episodes, the
model does not actually play this well consistently 😅

## Note

The model are trained over 3000 episodes played, with 10 epoch per episode.
After some testing, I found that the best batch size is the same size as the
episode's length, so basically performs one parameter update per epoch
with all the data in one episode. I've also limited the map pool size to 1000,
so there can only be 1000 different tracks during training. The final model
is able to attain an average score of around 600 to 700.

I've also tried out different hyperparameters, and observed that the
biggest factor that affects the training performance is the batch size
and map pool size. So far I've tested a variety of batch sizes but none of
them comes close to what I've end up with, which is as long as the episode's length.
I've also tried changing the map pool size to 10000 and unlimited, and I was able to
see progress with the 10000 map pool size but the performance was not as good as 1000,
and as for unlimited map pool size, the model didn't improve at all, at least with only
3000 episodes of training.

Below is a plot of total score vs episode during training, the gray line is
the model trained with 1000 map pool size, the yellow line is 10000, and the
cyan line is unlimited. I didn't finish all 3000 episodes of training
with the unlimited one because its obviously not improving.

![comparison](https://user-images.githubusercontent.com/20783502/216543974-4406aa64-fb69-46ab-af6b-51ef29320dca.png)

As the plot shows, lower map pool sizes tends to perform better, but I've only
tested three different sizes so it might be inaccurate.

I've save the tensorboards in the [runs](./runs/) directory, use
`tensorboard --logdir=runs` to see more plots such as loss vs episode.

My guesses are that lower map pool sizes are easier to train with less amount
of episodes, but the higher sizes might eventually performs better with more
episodes of training, but it remains to be tested.

## Todo

* Further improve the model (e.g. try out different model architecture, or train
the model for longer than just 3000 episodes)