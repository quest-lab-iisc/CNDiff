from models.diffusion_model import nonlinear_conditional_ddpm


def build_model(config):

    # Build diffusion model
    if config.diff.model_type == "CN_Diff":
        model = nonlinear_conditional_ddpm(config)
    else:
        assert True, "Kindly specify the correct diffusion model"

    print("Done Building Model")

    return model.to(config.train.device)

    
    
    
    


