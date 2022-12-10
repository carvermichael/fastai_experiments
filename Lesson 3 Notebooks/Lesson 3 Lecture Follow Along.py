#!/usr/bin/env python

# ## Lesson 3 Lecture Follow Along and Experimentation

# In[1]:

import matplotlib.pyplot as plt
import torch

plt.rc("figure", dpi=90)

def plot_function(f, title=None, min=-2.1, max=2.1, color='r', ylim=None):
	x = torch.linspace(min, max, 100)[:, None]
	if ylim:
		plt.ylim(ylim)
	plt.plot(x, f(x), color)
	if title is not None:
		plt.title(title)

# In[3]:

def f(x):
	return 3 * x**2 + 2 * x + 1

plot_function(f, "$3x^2 + 2x + 1$")

# In[4]:


def quad(a, b, c, x):
	return a * x**2 + b * x + c


# In[5]:

quad(3, 2, 1, 1.5)

# In[6]:

from functools import partial


def mk_quad(a, b, c):
	return partial(quad, a, b, c)


# In[7]:

f = mk_quad(3, 2, 1)
f(1.5)

# In[8]:

plot_function(f)

# In[9]:

import numpy as np
from numpy.random import normal, seed, uniform

np.random.seed(42)


def noise(x, scale):
	return normal(scale=scale, size=x.shape)


def add_noise(x, mult, add):
	return x * (1 + noise(x, mult)) + noise(x, add)


# In[10]:

import torch

x = torch.linspace(-2, 2, steps=20)[:, None]
y = add_noise(f(x), 0.3, 2.5)
plt.scatter(x, y)

# ## Working through the above functions

# In[11]:

linspaceResult = torch.linspace(-20, 20, 10)
print(linspaceResult)

# In[12]:

linTransposed = linspaceResult[:, None]
print(linTransposed)

# In[13]:

import torch

ones = torch.ones(5)
print(ones)

norm = normal(ones)
print(norm)

zeros = torch.zeros(5)
print(zeros)

norm = normal(zeros, 0.5)
print(norm)
norm = normal(zeros, 0.5, [8, 2, 5])
print(norm)

# ## Moving on...

# In[14]:

from ipywidgets import interact


@interact(a=1.5, b=1.5, c=1.5)
def plot_quad(a, b, c):
	plt.scatter(x, y)
	plot_function(mk_quad(a, b, c), ylim=(-3, 12))


# ## Loss Function

# In[15]:


def mse(x, y):
	return (x - y).square().mean().sqrt()

# In[20]:


@interact(a=1.5, b=1.5, c=1.5)
def plot_quad_loss(a, b, c):
	quad = mk_quad(a, b, c)
	plt.scatter(x, y)
	loss = mse(quad(x), y)
	plot_function(quad, ylim=(-3, 12), title=f"MSE: {loss:.2f}")


# ## Automating Loss Optimization

# In[19]:


def quad_mse(params):
	f = mk_quad(*params)
	return mse(f(x), y)


# In[21]:

quad_mse([1.5, 1.5, 1.5])

# In[22]:

abc = torch.tensor([1.5, 1.5, 1.5])
abc.requires_grad_()

# In[23]:

loss = quad_mse(abc)
print(loss)

# In[24]:

loss.backward()

# In[25]:

print(abc.grad)

# In[26]:

with torch.no_grad():
	abc -= abc.grad * 0.01
	loss = quad_mse(abc)

print(f'loss={loss:.2f}')

# In[28]:

for i in range(5):
	loss = quad_mse(abc)
	loss.backward()
	with torch.no_grad():
		abc -= abc.grad * 0.01
	print(f'step={i}, loss={loss:.2f}')

# In[29]:


def rectified_linear(m, b, x):
	y = m * x + b
	return torch.clip(y, 0.)


# In[30]:

plot_function(partial(rectified_linear, 1, 1))

# In[31]:


@interact(m=1.5, b=1.5)
def plot_relu(m, b):
	plot_function(partial(rectified_linear, m, b))


# In[33]:


def double_relu(m1, b1, m2, b2, x):
	return rectified_linear(m1, b1, x) + rectified_linear(m2, b2, x)


# In[36]:


@interact(m1=1.5, b1=1.5, m2=1.5, b2=1.5)
def plot_relu(m1, b1, m2, b2):
	plot_function(partial(double_relu, m1, b1, m2, b2))
