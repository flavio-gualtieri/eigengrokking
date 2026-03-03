import torch



def generate_fgsm_adversarial_examples(model, dataset, epsilon, device, batch_size=50):
    model.eval()
    adv_examples = []
    labels = []
    
    # use CrossEntropyLoss for adversarial attack. standard for classification
    loss_fn = torch.nn.CrossEntropyLoss()
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        # enable grad computation for input
        x.requires_grad = True
        
        # forward
        outputs = model(x)
        loss = loss_fn(outputs, y)
        
        # zero gradients
        model.zero_grad()
        
        # backpass to compute gradients w.r.t. input
        loss.backward()
        
        # get grad
        grad = x.grad.data
        
        # generate adversarial example
        adv_x = x + epsilon * grad.sign()
        
        # clamp to valid range (assuming input is in [0, 1])
        adv_x = torch.clamp(adv_x, 0, 1)
        
        adv_examples.append(adv_x.detach().cpu())
        labels.append(y.detach().cpu())
    
    adv_examples = torch.cat(adv_examples, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return adv_examples, labels