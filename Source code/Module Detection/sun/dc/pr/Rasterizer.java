package sun.dc.pr;
 
 import  sun.dc.path.*;
 import sun.java2d.Disposer;
 import sun.java2d.DisposerRecord;
 
 public class Rasterizer
 {
     static public final int
         // usage
         EOFILL          =  1,
         NZFILL          =  2,
         STROKE          =  3,
 
         // caps, corners
         ROUND           = PathStroker.ROUND,
         SQUARE          = PathStroker.SQUARE,
         BUTT            = PathStroker.BUTT,
         BEVEL           = PathStroker.BEVEL,
         MITER           = PathStroker.MITER,
 
         // tile size
         TILE_SIZE       = 1 << PathFiller.tileSizeL2S,
         TILE_SIZE_L2S   = PathFiller.tileSizeL2S,
 
         // implementation limits
         MAX_ALPHA       = PathFiller.MAX_PATH,
 
         MAX_MITER       = 10,
         MAX_WN          = 63,
 
         // tile states
         TILE_IS_ALL_0   = PathFiller.TILE_IS_ALL_0,
         TILE_IS_ALL_1   = PathFiller.TILE_IS_ALL_1,
         TILE_IS_GENERAL = PathFiller.TILE_IS_GENERAL;
 
     private static final int    BEG             = 1,
                                 PAC_FILL        = 2,
                                 PAC_STROKE      = 3,
                                 PATH            = 4,
                                 SUBPATH         = 5,
                                 RAS             = 6;
     private int                 state;
 
     private PathFiller          filler;
     private PathStroker         stroker;
     private PathDasher          dasher;
     private PathConsumer        curPC;                                  
 
     /*
      * This class ensures that the 3 Consumer objects are all
      * disposed when the Rasterizer falls on the floor now that
      * they are no longer individually finalizable.
      */
     private static class ConsumerDisposer implements DisposerRecord {
         PathConsumer filler;
         PathConsumer stroker;
         PathConsumer dasher;
 
         public ConsumerDisposer(PathConsumer filler,
                                 PathConsumer stroker,
                                 PathConsumer dasher)
         {
             this.filler = filler;
             this.stroker = stroker;
             this.dasher = dasher;
         }
 
         public void dispose() {
             filler.dispose();
             stroker.dispose();
             dasher.dispose();
         }
     }
 
     public Rasterizer() {
         state   = BEG;
         filler  = new PathFiller();
         stroker = new PathStroker(filler);
         dasher  = new PathDasher(stroker);
         Disposer.addRecord(this, new ConsumerDisposer(filler, stroker, dasher));
     }
 
     public void setUsage(int usage) throws PRError {
         if (state != BEG) {
             throw new PRError(PRError.UNEX_setUsage);
         }
         if (usage == EOFILL) {
             filler.setFillMode(PathFiller.EOFILL);
             curPC = filler;
             state = PAC_FILL;
         } else if (usage == NZFILL) {
             filler.setFillMode(PathFiller.NZFILL);
             curPC = filler;
             state = PAC_FILL;
         } else if (usage == STROKE) {
             curPC = stroker;
             filler.setFillMode(PathFiller.NZFILL);
             stroker.setPenDiameter((float)1.0);
             stroker.setPenT4(null);
             stroker.setCaps(ROUND);
             stroker.setCorners(ROUND, 0);
             state = PAC_STROKE;
         } else {
             throw new PRError(PRError.UNK_usage);
         }
 
     }
 
     public void setPenDiameter(float d) throws PRError {
         if (state != PAC_STROKE) {
             throw new PRError(PRError.UNEX_setPenDiameter);
         }
         stroker.setPenDiameter(d);
         /* state   = PAC_STROKE;        // in same state */
     }
 
     public void setPenT4(float[] t4) throws PRError {
         if (state != PAC_STROKE) {
             throw new PRError(PRError.UNEX_setPenT4);
         }
         stroker.setPenT4(t4);
         /* state   = PAC_STROKE;        // in same state */     
     }
 
     public void setPenFitting(float unit, int mindiameter) throws PRError {
         if (state != PAC_STROKE) {
             throw new PRError(PRError.UNEX_setPenFitting);
         }
         stroker.setPenFitting(unit, mindiameter);
         /* state   = PAC_STROKE;        // in same state */     
     }
 
     public void setPenDisplacement(float dx, float dy) throws PRError {
         if (state != PAC_STROKE) {
             throw new PRError(PRError.UNEX_setPenDisplacement);
         }
         float[] t6 = {(float)1.0, (float)0.0, (float)0.0, (float)1.0, dx, dy};
         stroker.setOutputT6(t6);
         /* state   = PAC_STROKE;        // in same state */     
     }
 
     public void setCaps(int caps) throws PRError {
         if (state != PAC_STROKE) {
             throw new PRError(PRError.UNEX_setCaps);
         }
         stroker.setCaps(caps);
         /* state   = PAC_STROKE;        // in same state */
     }
 
     public void setCorners(int corners, float miter) throws PRError {
         if (state != PAC_STROKE) {
             throw new PRError(PRError.UNEX_setCorners);
         }
         stroker.setCorners(corners, miter);
         /* state   = PAC_STROKE;        // in same state */
     }
 
     public void setDash(float[] dash, float offset) throws PRError {
         if (state != PAC_STROKE) {
             throw new PRError(PRError.UNEX_setDash);
         }
         dasher.setDash(dash, offset);
         curPC = dasher;                 // if the previous call is successful
         /* state = PAC_STROKE;          // in same state */
     }
 
     public void setDashT4(float[] dasht4) throws PRError {
         if (state != PAC_STROKE) {
             throw new PRError(PRError.UNEX_setDashT4);
         }
         dasher.setDashT4(dasht4);
         /* state   = PAC_STROKE;        // in same state */
     }
 
     public void beginPath(float[] box) throws PRError {
         beginPath();
     }
     public void beginPath() throws PRError {
         if (state != PAC_FILL && state != PAC_STROKE) {
             throw new PRError(PRError.UNEX_beginPath);
         }
         try {
             curPC.beginPath();
             state = PATH;
         } catch (PathError e) {
             throw new PRError(e.getMessage());
         }
     }
 
     public void beginSubpath(float x0, float y0) throws PRError {
         if (state != PATH && state != SUBPATH) {
             throw new PRError(PRError.UNEX_beginSubpath);
         }
         try {
             curPC.beginSubpath(x0, y0);
             state = SUBPATH;
         } catch (PathError e) {
             throw new PRError(e.getMessage());
         }
     }
 
     public void appendLine(float x1, float y1) throws PRError {
         if (state != SUBPATH) {
             throw new PRError(PRError.UNEX_appendLine);
         }
         try {
             curPC.appendLine(x1, y1);
         } catch (PathError e) {
             throw new PRError(e.getMessage());
         }
     }
 
     public void appendQuadratic(float xm, float ym, float x1, float y1)
         throws PRError {
         if (state != SUBPATH) {
             throw new PRError(PRError.UNEX_appendQuadratic);
         }
         try {
             curPC.appendQuadratic(xm, ym, x1, y1);
         } catch (PathError e) {
             throw new PRError(e.getMessage());
         }
     }
 
     public void appendCubic(float xm, float ym, float xn, float yn, float x1, float y1)
         throws PRError {
         if (state != SUBPATH) {
             throw new PRError(PRError.UNEX_appendCubic);
         }
         try {
             curPC.appendCubic(xm, ym, xn, yn, x1, y1);
         } catch (PathError e) {
             throw new PRError(e.getMessage());
         }
     }
 
     public void closedSubpath() throws PRError {
         if (state != SUBPATH)
             throw new PRError(PRError.UNEX_closedSubpath);
         try {
             curPC.closedSubpath();
         } catch (PathError e) {
             throw new PRError(e.getMessage());
         }
     }
 
     public void endPath() throws PRError, PRException {
         if (state != PATH && state != SUBPATH)
             throw new PRError(PRError.UNEX_endPath);
         try {
             curPC.endPath();
             state = RAS;
         } catch (PathError e) {
             throw new PRError(e.getMessage());
         } catch (PathException e) {
             throw new PRException(e.getMessage());
         }
     }
 
     public void useProxy(FastPathProducer proxy) throws PRError, PRException {
         if (state != PAC_FILL && state != PAC_STROKE) {
             throw new PRError(PRError.UNEX_useProxy);
         }
         try {
             curPC.useProxy(proxy);
             state = RAS;
         } catch (PathError e) {
             throw new PRError(e.getMessage());
         } catch (PathException e) {
             throw new PRException(e.getMessage());
         }
     }
 
     public void getAlphaBox(int[] box) throws PRError {
         filler.getAlphaBox(box);
     }
 
     public void setOutputArea(float x0, float y0, int w, int h)
                         throws PRError, PRException {
         filler.setOutputArea(x0, y0, w, h);
     }
 
     public int getTileState() throws PRError {
         return filler.getTileState();   /* return current tile state */
     }
 
     public void writeAlpha(byte[] alpha, int xstride, int ystride, int pix0offset)
                         throws PRError, PRException, InterruptedException {
         filler.writeAlpha(alpha, xstride, ystride, pix0offset);
     }
 
     public void writeAlpha(char[] alpha, int xstride, int ystride, int pix0offset)
                         throws PRError, PRException, InterruptedException {
         filler.writeAlpha(alpha, xstride, ystride, pix0offset);
     }
 
     public void nextTile() throws PRError {
         filler.nextTile();
     }
     
     public void reset() {
         state = BEG;
         filler.reset();
         stroker.reset();
         dasher.reset();
     }
 }


