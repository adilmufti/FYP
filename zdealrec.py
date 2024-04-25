'''
These testing functions were called on the model, after
fine-tuning on different permutations of training samples
'''


import torch


def hvp(ys, xs, vs):
    grads = torch.autograd.grad(ys, xs, grad_outputs=vs, only_inputs=True, retain_graph=True)
    return torch.cat([grad.view(-1) for grad in grads])


def influence_function(model, loss_fn, train_loader, test_loader, device):
    """Compute the influence scores using Hessian-vector products."""
    model.eval()
    influence_scores = []

    for test_inputs, test_targets in test_loader:
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        test_loss = loss_fn(model(test_inputs), test_targets)

        test_grad = torch.autograd.grad(test_loss, model.parameters())
        test_grad_vec = torch.cat([grad.view(-1) for grad in test_grad])

        influences = []

        for train_inputs, train_targets in train_loader:
            train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)

            train_outputs = model(train_inputs)
            train_loss = loss_fn(train_outputs, train_targets)

            grad_train_loss = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)
            grad_train_loss_vec = torch.cat([grad.view(-1) for grad in grad_train_loss])

            hv = hvp(train_loss, model.parameters(), grad_train_loss_vec)

            influence = -torch.dot(test_grad_vec, hv).item()
            influences.append(influence)

        influence_scores.append(influences)

    return influence_scores



def calculate_effort_scores(model, loss_fn, dataloader, device):
    # Called with un-fine-tuned model
    model.eval()
    effort_scores = []

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
        # grad_norm_working = torch.norm(torch.stack([param.grad.flatten() for param in model.parameters()])).item()

        effort_scores.append(grad_norm)
        # effort_scores.append(grad_norm_working)

    return effort_scores



