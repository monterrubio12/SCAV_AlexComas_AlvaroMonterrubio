class exercises:
    def RGBtoYUV(r,g,b):
        Y= 0.257*R + 0.504*G + 0.098*B + 16
        U= -0.148*R - 0.291*G +0.439*B + 128
        V= 0.439*R - 0.368*G - 0.071*B + 128
        return Y,U,V

    def YUVtoRGB(Y,U,V)
        R= 1.164*(Y-16)

    #CAMBIOS ALEXX