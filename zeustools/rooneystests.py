from zeustools.rooneystools import nd_mad
import numpy as np

def test_nd_mad():
    assert nd_mad(np.array([1,8000,1,2]),0)==0.5
    assert nd_mad(np.array([1,1,1,387]),0)==0.0
    test_2d_1 = np.zeros((100,2))
    test_2d_1[50,1]=23
    test_2d_1[33,1]=-39
    test_2d_1[90,0]=7
    output_2d_1=nd_mad(test_2d_1,0)
    print(output_2d_1)
    assert np.allclose(output_2d_1,np.zeros(2))
    output_2d_2=nd_mad(test_2d_1,1)
    correct_output_2d_2=np.zeros(100)
    correct_output_2d_2[33]=39/2
    correct_output_2d_2[50]=23/2
    correct_output_2d_2[90]=3.5
    assert np.allclose(output_2d_2,correct_output_2d_2)
    test_3d=np.array([[[ 1.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 1.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 1.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 1.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 1.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]],

       [[ 2.        ,  2.99880138,  0.        ],
        [ 0.        , -0.1854465 ,  0.        ],
        [ 2.        , -5.82443017,  0.        ],
        [ 0.        , -5.27777733,  0.        ],
        [ 2.        , -6.1162822 ,  0.        ],
        [ 0.        ,  3.15357023,  0.        ],
        [ 2.        ,  4.37515736,  0.        ],
        [ 0.        , -9.14755405,  0.        ],
        [ 2.        , -8.94759466,  0.        ],
        [ 0.        ,  7.46094059,  0.        ]],

       [[ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]]])
    correct_output_3d=np.array([[0.5       , 0.        , 0.        ],
       [1.        , 5.80779772, 0.        ],
       [0.        , 0.        , 0.        ]])
    assert np.allclose(nd_mad(test_3d,1),correct_output_3d)

