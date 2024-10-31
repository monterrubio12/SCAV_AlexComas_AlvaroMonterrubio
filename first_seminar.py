class exercises:
    def RGBtoYUV(r,g,b):
        Y= 0.257*g + 0.504*g + 0.098*b + 16
        U= -0.148*g - 0.291*g +0.439*b + 128
        V= 0.439*g - 0.368*g - 0.071*b + 128
        return Y,U,V

    def YUVtoRGB(Y,U,V)
        R= 1.164*(Y-16)

    #cambios