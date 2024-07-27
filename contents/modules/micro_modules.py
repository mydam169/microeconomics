import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns


def load_viz_settings():
    sns.set_style('whitegrid')
    plot_params = {'grid.linestyle' : ':', 
                  'grid.color' : 'grey', 
                  'grid.linewidth' : '1', 
                  'font.family' : 'arial', 
                  }
    for kk, vv in plot_params.items():
        plt.rcParams[kk] = vv

def cobb_douglas(x, y, a, b, A=1):
    return A * (x ** a) * (y ** b)

def plotIC(x, y, alph, bet, A=1, name='', lab=False, levels=np.arange(.5,  2, .25)):
    
    X, Y = np.meshgrid(x, y)
    U = cobb_douglas(X, Y, alph, bet, A=A)
    # fig = plt.figure(figsize=plt.figaspect(.5))
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(name)
    
    ax = fig.add_subplot(1, 2, 1)
    ax = plt.contour(X, Y, U, levels=levels, cmap='plasma', linewidth=2)
    if lab:
        plt.clabel(ax, inline=True, fontsize=10)
    # plt.colorbar(ax)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_surface(X, Y, U, edgecolor='white', 
                    cmap=cm.plasma,
                    lw=.5, rstride=8, cstride=8, 
                    alpha=.8
                   )
    plt.show()


def plotIC_only(x, y, alph, bet, levels, A=1, name='', lab=False):
    
    X, Y = np.meshgrid(x, y)
    U = cobb_douglas(X, Y, alph, bet, A=A)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    ax = plt.contour(X, Y, U, levels=levels, cmap='plasma')
    if lab:
        plt.clabel(ax, inline=True, fontsize=10)
    plt.show()

def curve(x):
    return 1 / x


def plotIC_thick():
    x = np.linspace(0.1, 4, 500)
    y = curve(x)

    plt.plot(x, y, 'k-', linewidth=50, alpha=0.25, label='Thick IC')
    plt.plot(x, y, 'k-', linewidth=2)

    # Set labels and title
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Thick IC: a violation of local nonsatiation')


    # Set axis limits
    plt.xlim(0, 2.5)
    plt.ylim(0, 2.5)
    # suppress ticks
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.grid(False)
    plt.savefig('images/thick_IC.png')
    plt.show()

def plotIC_curve():
    x = np.linspace(.1, 4, 500)
    y = curve(x)
    plt.plot(x, y, color='green')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('An indifference curve')

    # Add text annotations
    plt.text(1.5, 2, 'Upper contour set (UCS)\n$\{y \in\mathbb{R}^2_+: y \succeq x\}$', fontsize=11)
    plt.text(0.25, 0.5, 'Lower contour set (LCS)\n$\{y \in\mathbb{R}^2_+: x \succeq y\}$', fontsize=10)
    plt.text(1.5, 0.75, 'Indifference set\n$\{y \in\mathbb{R}^2_+: y \sim x\}$', fontsize=12, color='green')

    # Add a point X on the curve
    plt.plot(1, 1, 'ko')
    plt.text(1.1, 1.1, 'X', fontsize=12)

    # Set axis limits
    plt.xlim(0, 2.5)
    plt.ylim(0, 2.5)

    # suppress ticks
    plt.xticks([])
    plt.yticks([])

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Show the plot
    plt.grid(False)
    plt.savefig('images/IC_curve.png')
    plt.show()

def plot_convex_UCS():
    x = np.linspace(0.1, 4, 200)
    y1 = 1 / x
    y2 = np.ones(len(x)) * np.max(x)

    plt.fill_between(x, y1, y2, alpha=0.5, color='gray')
    plt.plot(x, y1)
    plt.plot(x, y2, color='white')

    # plt.xlabel('$x$')
    # plt.ylabel('$y$')

    # adding points
    plt.plot(0.5, 2, 'ko')
    plt.text(0.55, 2.05, 'x', fontsize=12)
    plt.plot(2, 0.5, 'ko')
    plt.text(2.05, 0.55, 'y', fontsize=12)
    plt.plot(1., 1., 'ko')
    plt.text(1.05, 1.05, 'z', fontsize=12)

    # line between two points
    plt.plot([.5, 2], [2, .5], 'k-')
    plt.text(1.3, 1.3, '$\\alpha x + (1 - \\alpha) y$', fontsize=14)
    # plt.text(1.6, 1.8, "UCS", fontsize=14)

    plt.xlim([.25, 2.5])
    plt.ylim([.25, 2.5])
    plt.xticks([])
    plt.yticks([])
    plt.savefig('images/convex_UCS')

    plt.show()

def plotMRS():
    x = np.linspace(0.1, 5, 400)
    y = 2 / x
    plt.plot(x, y)

    # points
    plt.plot(.5, 4., 'ko')
    plt.plot(1., 2., 'ko')
    plt.plot(1.5, 2 / 1.5, 'ko')

    plt.vlines(x=0.5, ymin=2, ymax=4, linestyle='--', color='gray')
    plt.vlines(x=1., ymin=2/1.5, ymax=2, linestyle='--', color='gray')
    plt.hlines(y=2, xmin=0.5, xmax=1, linestyle='--', color='gray')
    plt.hlines(y=2/1.5, xmin=1, xmax=1.5, linestyle='--', color='gray')

    # annotations
    plt.text(0.55, 4.05, 'A', fontsize=14)
    plt.text(1., 2.1, 'B', fontsize=14)
    plt.text(1.5, 1.5, 'C', fontsize=14)
    plt.text(0.35, 3, '$\\Delta y$')
    plt.text(0.6, 2.1, '$\\Delta x$')
    plt.text(0.85, 1.6, "$\\Delta y'$")
    plt.text(1.2, 1.1, '$\\Delta x$')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, 2.5])
    plt.ylim([0, 4.5])

    plt.xticks([])
    plt.yticks([])

    plt.show()


    
    
    