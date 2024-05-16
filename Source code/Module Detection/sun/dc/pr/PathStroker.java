package sun.dc.pr;

import  sun.dc.path.*;

public class PathStroker implements PathConsumer {

//  ____________________________________________________________________
//
//  PUBLIC CONSTANTS
//  ____________________________________________________________________

     public static final int     ROUND           = 10,   // caps, corners
                                SQUARE          = 20,
                               BUTT            = 30,
                                BEVEL           = 40,
                                MITER           = 50;

//  ____________________________________________________________________
//
//  CONSTRUCTOR and DESTRUCTOR
//  ____________________________________________________________________

    private PathConsumer dest;

     public      PathStroker(PathConsumer dest) {
        this.dest = dest;
        cInitialize(dest);
        reset();
    }
    public native void dispose();
     protected static void classFinalize() throws Throwable {
        cClassFinalize();
    }

    public PathConsumer getConsumer() {
        return dest;
    }

//  ____________________________________________________________________
//
//  PATH STROKER METHODS
//  ____________________________________________________________________

    public native void  setPenDiameter(float d) throws PRError;

    public native void  setPenT4(float[] t4) throws PRError;

    public native void  setPenFitting(float unit, int mindiameter) throws PRError;

    public native void  setCaps(int caps) throws PRError;

    public native void  setCorners(int corners, float miter) throws PRError;

    public native void  setOutputT6(float[] t6) throws PRError;

    public native void  setOutputConsumer(PathConsumer dest) throws PRError;

    public native void  reset();


//  ____________________________________________________________________
//
//  PATHCONSUMER IMPLEMENTATION
//  ____________________________________________________________________

    public native void  beginPath() throws PathError;

    public native void  beginSubpath(float x0, float y0)throws PathError;

    public native void  appendLine(float x1, float y1)  throws PathError;

    public native void  appendQuadratic(float xm, float ym, float x1, float y1)
                                throws PathError;

    public native void  appendCubic(float xm, float ym, float xn, float yn,
                                    float x1, float y1) throws PathError;

    public native void  closedSubpath() throws PathError;

    public native void  endPath() throws PathError, PathException;

     public void useProxy(FastPathProducer proxy) throws PathError, PathException {
        proxy.sendTo(this);
    }

    public native long  getCPathConsumer();

//  ____________________________________________________________________
//
//  CONNECTIONS to dcPathStroker in C
//  ____________________________________________________________________
    static {
        java.security.AccessController.doPrivileged(
                  new sun.security.action.LoadLibraryAction("dcpr"));
        cClassInitialize();
    }
    private static native void  cClassInitialize();
    private static native void  cClassFinalize();

    private        long cData;
    private native void cInitialize(PathConsumer dest);
}

