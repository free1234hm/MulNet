package sun.dc.pr;
 
 import sun.dc.path.*;
 
 public class PathDasher implements PathConsumer {
 
 //  ____________________________________________________________________________
 //
 //  PUBLIC CONSTANTS
 //  ____________________________________________________________________________
 
 //  ____________________________________________________________________________
 //
 //  CONSTRUCTOR and DESTRUCTOR
 //  ____________________________________________________________________________
 
     private PathConsumer dest;
 
     public PathDasher(PathConsumer dest) {
         this.dest = dest;
         cInitialize(dest);
        reset();
     }
     public native void  dispose();
     protected static void       classFinalize() throws Throwable {
         cClassFinalize();
     }
 
     public PathConsumer getConsumer() {
         return dest;
     }
 
 //  ____________________________________________________________________________
 //
 //  PATH DASHER METHODS
 //  ____________________________________________________________________________
 
     private static final float  TOP_MAX_MIN_RATIO = 100F;
 
     public native  void setDash(float[] dash, float offset) throws PRError;
 
     public native  void setDashT4(float[] t4) throws PRError;
 
     public native void  setOutputT6(float[] t6) throws PRError;
 
     public native void  setOutputConsumer(PathConsumer dest) throws PRError;
 
     public native  void reset();
 
 //  ____________________________________________________________________________
 //
 //  PATHCONSUMER IMPLEMENTATION
 //  ____________________________________________________________________________
 
     public native  void beginPath() throws PathError;
 
     public native  void beginSubpath(float x0, float y0) throws PathError;
 
     public native  void appendLine(float x1, float y1) throws PathError;
 
     public native  void appendQuadratic(float xm, float ym, float x1, float y1)
                                 throws PathError;
 
     public native  void appendCubic(float xm, float ym, float xn, float yn,
                                     float x1, float y1) throws PathError;
 
     public native  void closedSubpath() throws PathError;
 
     public native void  endPath() throws PathError, PathException;
 
     public void useProxy(FastPathProducer proxy) throws PathError, PathException {
         proxy.sendTo(this);
     }
 
     public native long  getCPathConsumer();
     
 //  ____________________________________________________________________________
 //
 //  CONNECTIONS to dcPathDasher in C
 //  ____________________________________________________________________________
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
