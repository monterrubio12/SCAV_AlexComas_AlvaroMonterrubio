class exercises:
   def RGBtoYUV(r,g,b):
       Y= 0.257*r + 0.504*g + 0.098*b + 16
       U= -0.148*r - 0.291*g +0.439*b + 128
       V= 0.439*r - 0.368*g - 0.071*b + 128
       return Y,U,V


   def YUVtoRGB(Y,U,V):
       R= 1.164*(Y-16) + 2.018*(U-128)
       G= 1.164*(Y-16) - 0.813*(V-128) - 0.391*(U-128)
       B= 1.164*(Y-16) + 1.596*(V-128)
       return R,G,B