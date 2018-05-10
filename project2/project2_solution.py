def main(): # function to demonstrate various adv attacks on an image
    import foolbox
    import keras
    import numpy as np
    from keras.applications.resnet50 import ResNet50
    from foolbox.criteria import Misclassification
    from keras.preprocessing import image
    from keras.applications.resnet50 import preprocess_input, decode_predictions
#   %matplotlib inline
    import matplotlib
    matplotlib.use('TKAgg')
    import matplotlib.pyplot as plt
    import numpy as np
    from keras import backend as K
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    K.set_image_dim_ordering('tf')  # this is very important!
    from PIL import Image
    import glob

    # instantiate model
    keras.backend.set_learning_phase(0)
    kmodel = ResNet50(weights='imagenet')
    preprocessing = (np.array([104, 116, 123]), 1)
    fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255) )#, preprocessing=preprocessing)

    # get source image and label
    image, label = foolbox.utils.imagenet_example()
    criterion = foolbox.criteria.Misclassification()

    image_list = []
    for filename in glob.glob('./*.jpg'):  # assuming jpg
        im = Image.open(filename)
        image_list.append(im)

    image = np.array(image_list[0])
    fmodel.predictions(image).argmax()
 #   plt.figure()
#    plt.imshow(image)
#    plt.show()
#    plt.savefig('test.pdf')

#    attack = foolbox.attacks.SaliencyMapAttack(fmodel,criterion=criterion)

    attack_dict = {
            #  Gradient-based Attacks
            'AGNA':foolbox.attacks.AdditiveGaussianNoiseAttack,
            'AUNA':foolbox.attacks.AdditiveUniformNoiseAttack,
            'BUNA':foolbox.attacks.BlendedUniformNoiseAttack,  # very fast.
            'CRA':foolbox.attacks.ContrastReductionAttack,
            'GA':foolbox.attacks.GradientAttack,
            'GBA':foolbox.attacks.GaussianBlurAttack,
            'GSA':foolbox.attacks.GradientSignAttack,
       #     'IGSA':foolbox.attacks.IterativeGradientSignAttack,
        #    'IGA':foolbox.attacks.IterativeGradientAttack,
            'LBFGSA':foolbox.attacks.LBFGSAttack,
#            'SLSQPA':foolbox.attacks.SLSQPAttack,
            'SMA':foolbox.attacks.SaliencyMapAttack,
#            'DFA':foolbox.attacks.DeepFoolAttack,   # needs modification

            #  Score-based attacks
#            'SPA':foolbox.attacks.SinglePixelAttack, # thinks HPC dim is image columns dim - fix it
#            'LSA':foolbox.attacks.LocalSearchAttack, # thinks HPC dim is image columns dim - fix it
#            'ALBFGSA':foolbox.attacks.ApproximateLBFGSAttack,

            #  Decision-based attacks
#            'BA':foolbox.attacks.BoundaryAttack,  # works in n-steps to reduce perturbation size, very slow with large # of steps
            'SPNA':foolbox.attacks.SaltAndPepperNoiseAttack,  # very fast.
#            'RA':foolbox.attacks.ResetAttack  # Starts with an adversarial and resets as many values as possible to the values of the original.
            }

    i = 1
    plt.figure(figsize=(4, 5))
    sindex = [1,2,3,4,5,11,12,13,14,15]
    for attack in attack_dict:
        try:
            image = np.array(image,dtype='int64')
            #label = kmodel.predict(np.expand_dims(image, 0)).argmax()
            label = fmodel.predictions(image).argmax()
            print('i is %d' %i)
            print('image label is %d' %label)

            selected_attack = attack_dict[attack](model=fmodel, criterion=criterion)

            adversarial = selected_attack(image, label)
            #adversarial = attack(image, label)
            label = fmodel.predictions(adversarial).argmax()
            print('adversarial label is %d' %label)

            idx = sindex[i-1] # the subplot index
#            plt.subplot(5, 4, ((i-1)*3)+1)
#            plt.title('Original')
#            plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
#            plt.axis('off')

            plt.subplot(4, 5, idx)
            plt.title(attack)
            plt.imshow(adversarial / 255)  # ::-1 to convert BGR to RGB
            plt.axis('off')

            plt.subplot(4, 5, idx+5)
            plt.title(attack + ' Pert.')
            difference = adversarial - image
            plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)

            plt.axis('off')
            i +=1
        except TypeError:
            print('\nCould not create perturbation for sample: %d' %i)
    plt.show()


    plt.title(attack.name() + ' results')
    plt.savefig( attack.name() + '.pdf')
    plt.close()

if __name__== '__main__':
    main()
