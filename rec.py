from dust3r.utils.image import load_images
from dust3r.inference import inference, load_model
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


if __name__ =="__main__":
    model_pth='/Users/ekole/Dev/dust3r/checkpoints/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth'
    device='cpu' #cuda(nvidia)  or mps(apple silicon)
    batch_size=1
    scheduler='cosine'
    lr=0.01
    niter=300

    model=load_model(model_pth, device)

    images=load_images(['/Users/ekole/Dev/gut_slam/gut_images/image1.jpeg', '/Users/ekole/Dev/gut_slam/gut_images/image2.jpeg'], size=512)
    pairs=make_pairs(images, scene_graph='complete', prefilter=None,symmetrize=True)
    output=inference(pairs, model, device, batch_size)

    #raw outputs of dust3r

    view1, pred1=output['view1'], output['pred1']       #here view1, pred1,view 2, pred2 are dicts of len(2)
    view2, pred2=output['view2'], output['pred2']       #due to symetrize=True, we have (im1,im2) and (im2,im1) pairs

    scene=global_aligner(output,device=device,mode=GlobalAlignerMode.PointCloudOptimizer)
    loss=scene.compute_global_alignment(init='mst', niter=niter, lr=lr, schedule=scheduler)

    #retrieve useful info from scene

    imgs=scene.imgs
    focals=scene.get_focals()
    poses=scene.get_im_poses()
    pts3d=scene.get_pts3d()
    confi_mask=scene.get_masks()

    #visualise reconstruction
    #scene.show()

    print('focus:', focals)
    print('poses:', poses)
    print('3d points:', pts3d)

