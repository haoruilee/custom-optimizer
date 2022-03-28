from functools import partial

def flatten(x):
    original_shape = x.shape
    return x.flatten(), partial(np.reshape, newshape=original_shape)

def unflatten_optimizer_step(step):
    """
    Wrap an optimizer step function that operates on flat 1D arrays
    with a version that handles trees of nested containers,
    i.e. (lists/tuples/dicts), with arrays/scalars at the leaves.
    """
    # 装饰后的 step 函数
    def _step(value_and_grad, x, itr, state=None, *args, **kwargs):
        _x, unflatten = flatten(x)
        
        def _value_and_grad(x):
            v, g = value_and_grad(unflatten(x))
            return v, flatten(g)[0]
        
        _next_x, _next_val, _next_g, _next_state = step(_value_and_grad, _x, itr, state=state, *args, **kwargs)
        return unflatten(_next_x), _next_val, _next_g, _next_state
    
    return _step



@unflatten_optimizer_step
def sgd_step(value_and_grad, x, itr, state=None, step_size=0.1, mass=0.9):
    # Stochastic gradient descent with momentum.
    velocity = state if state is not None else np.zeros(len(x))
    val, g = value_and_grad(x)
    velocity = mass * velocity - (1.0 - mass) * g
    x = x + step_size * velocity
    return x, val, g, velocity

@unflatten_optimizer_step
def rmsprop_step(value_and_grad, x, itr, state=None, step_size=0.1, gamma=0.9, eps=10**-8):
    # Root mean squared prop: See Adagrad paper for details.
    avg_sq_grad = np.ones(len(x)) if state is None else state
    val, g = value_and_grad(x)
    avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
    x = x - (step_size * g) / (np.sqrt(avg_sq_grad) + eps)
    return x, val, g, avg_sq_grad


@unflatten_optimizer_step
def adam_step(value_and_grad, x, itr, state=None, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """
    Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms.
    """
    m, v = (np.zeros(len(x)), np.zeros(len(x))) if state is None else state
    val, g = value_and_grad(x)
    m = (1 - b1) * g      + b1 * m    # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v    # Second moment estimate.
    mhat = m / (1 - b1**(itr + 1))    # Bias correction.
    vhat = v / (1 - b2**(itr + 1))
    x = x - (step_size * mhat) / (np.sqrt(vhat) + eps)
    return x, val, g, (m, v)


def _generic_sgd(method, loss, x0, callback=None, num_iters=200, state=None, history=False, **kwargs):
    """
    Generic stochastic gradient descent step.
    """
    step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[method]
    
    def loss_value_and_grad(loss, x):
        return loss.value(x), loss.gradient(x)
    value_and_grad = partial(loss_value_and_grad, loss)
    
    # Initialize outputs
    x, losses, grads = x0, [], []
    for itr in range(num_iters):
        x, val, g, state = step(value_and_grad, x, itr, state, **kwargs)
        losses.append(val)
        grads.append(g)

    if history:
        return x, losses, grads
    else:
        return x


# Define optimizers
sgd = partial(_generic_sgd, "sgd")
rmsprop = partial(_generic_sgd, "rmsprop")
adam = partial(_generic_sgd, "adam")

class MyLoss:
    def __init__(self, x, y):
    	"""
    	y = w @ x
		"""
        self.x = x
        self.y = y
        
    def value(self, w):
        return 0.5 * np.linalg.norm(y - w @ x)
    
    def gradient(self, w):
        return w @ x @ x.T - y @ x.T

m = 20
n = 10
N = 1000

x = np.random.random((n,N))
w = np.random.random((m,n))
y = w @ x

y += np.random.randn(*y.shape)*0.001

loss = MyLoss(x,y)

w0 = np.random.random(w.shape)

fig, ax = plt.subplots(figsize=(5,5))

_, history,_ = sgd(loss, w0,num_iters=200, history=True,step_size=0.001, mass=0.3)
plt.plot(history, label='sgd')

_, history,_ = adam(loss, w0,num_iters=200, history=True,step_size=0.1, b1=0.9, b2=0.999,)
plt.plot(history, label='adam')

_, history,_ = rmsprop(loss, w0,num_iters=200, history=True,step_size=0.01, gamma=0.9)
plt.plot(history, label='rmsprop')

ax.set_yscale('log')
plt.legend()
plt.xlabel('step')
plt.ylabel('loss')
plt.show()
