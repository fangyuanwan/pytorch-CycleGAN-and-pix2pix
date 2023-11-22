# import os
# from options.test_options import TestOptions
# from data import CreateDataLoader
# from models import create_model
# from util.visualizer import save_images
# from util import html


# if __name__ == '__main__':
#     opt = TestOptions().parse()
#     opt.nThreads = 1   # test code only supports nThreads = 1
#     opt.batchSize = 1  # test code only supports batchSize = 1
#     opt.serial_batches = True  # no shuffle
#     opt.no_flip = True  # no flip
#     opt.display_id = -1  # no visdom display
#     data_loader = CreateDataLoader(opt)
#     dataset = data_loader.load_data()
#     model = create_model(opt)
#     model.setup(opt)
#     # create website
#     web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
#     webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
#     # test
#     for i, data in enumerate(dataset):
#         if i >= opt.how_many:
#             break
#         model.set_input(data)
#         model.test()
#         visuals = model.get_current_visuals()
#         img_path = model.get_image_paths()
#         if i % 5 == 0:
#             print('processing (%04d)-th image... %s' % (i, img_path))
#         save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

#     webpage.save()

import torch
import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html

if __name__ == '__main__':
    torch.set_num_threads(1)  # Set the number of threads for PyTorch at the beginning

    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print(dataset)
    
    # First pass over the DataLoader
    for i, data in enumerate(dataset):
        images_A = data['A']  # Images from domain A
        images_B = data['B']  # Images from domain B
        paths_A = data['A_paths']  # File paths for images from domain A
        paths_B = data['B_paths']  # File paths for images from domain B

        # ... your code to process the images ...

        print(f"Batch {i+1}")
        print(f"Domain A Images Tensor Shape: {images_A.shape}")
        print(f"Domain B Images Tensor Shape: {images_B.shape}")
        # Optionally print paths if needed:
        # print(f"Domain A Image Paths: {paths_A}")
        # print(f"Domain B Image Paths: {paths_B}")
        print("-" * 30)
        
         # Break after the 5th iteration
        if i == 4:
            break
    
    # Reinitialize the DataLoader for the second pass
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    
    model = create_model(opt)
    model.setup(opt)
    
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    
    # Second pass over the DataLoader
    for i, data in enumerate(dataset):
        print(f"Processing batch {i}")
        if i >= opt.how_many:
            break
        
        try:
            model.set_input(data)
            model.test()  # This is where it seems to get stuck, so pay attention to the output
        except Exception as e:
            print(f"An error occurred: {e}")
            break  # Break out of the loop on error
        
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        
        if i % 5 == 0:
            print(f'processing ({i:04d})-th image... {img_path}')
        
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        
    webpage.save()
    print("Completed processing all batches.")
