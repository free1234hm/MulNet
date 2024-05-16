package sun.dc.pr;             
              
import sun.dc.path.*;             
              
public class PathFiller implements PathConsumer          
{              
//_____________________________________________________________________              
//              
// PUBLIC CONSTANTS            
// fill modes, tile states, sizes         
// CONSTRUCTOR             
// public constructor            
// INITIALIZATION             
// private state, fillmode and related methods and constants      
// public method setFillMode           
// DEFINITION OF OUTPUT AREA          
// private pathBox            
// public method getAlphaBox           
// private static class Run, related constants        
// private tileRuns array, etc          
// private output area positions, dimensions, etc        
// /*private*/ methods processToRunsArc1/2/3           
// private method runCheckForArcAppend,           
// private methods appendToRunARc1/2/3           
// private runsBuilder (RunsBuilder implements PathConsumer)         
// public method setOutputArea           
// TILE RETRIEVAL            
// private static class LeftSide          
// private lsAffects            
// public method nextTile           
// public method getTileState           
// private fastOutputPC (FastOutputPC implements PathConsumer)         
// private method sendTileToLLFiller           
// public methods writeAlpha           
// public method reset           
//_____________________________________________________________________              
              
              
// ___________________________________________________________________             
//              
// PUBLIC CONSTANTS            
// ___________________________________________________________________             
              
public static final int           
EOFILL = 1, //           
NZFILL = 2, // explicit in jc case       
              
MAX_PATH = 1000000, // maximum path coordinate allowed       
              
TILE_IS_ALL_0 = 0, // tile states         
TILE_IS_ALL_1 = 1,            
TILE_IS_GENERAL = 2;            
              
/* This field should be package private to avoid external modification */   
/* public */ static /* final */ int       
tileSizeL2S; // to be set by c filler at static init time   
              
private static /* final */ int         
tileSize; // to be set by c filler at static init time   
              
private static /* final */ float         
tileSizeF; // to be set by c filler at static init time   
              
public static final float           
maxPathF = (float)MAX_PATH;            
public static final boolean           
validLoCoord(float c) { return c >= -maxPathF; }       
public static final boolean           
validHiCoord(float c) { return c <= maxPathF; }       
              
// ___________________________________________________________________             
//              
// CONSTRUCTOR             
// ___________________________________________________________________             
              
public PathFiller() {            
cInitialize();              
reset();              
}              
public native void dispose();           
protected static void classFinalize() throws Throwable {        
cClassFinalize();              
}              
              
public PathConsumer getConsumer() {           
return null;             
}              
              
// ___________________________________________________________________             
//              
// INITIALIZATION             
// ___________________________________________________________________             
              
public native void setFillMode(int fillmode) throws PRError;        
              
// ___________________________________________________________________             
//              
// PATH DESCRIPTION            
// ___________________________________________________________________             
//              
public native void beginPath() throws PathError;         
              
public native void beginSubpath(float x0, float y0) throws PathError;      
              
public native void appendLine(float x1, float y1) throws PathError;      
              
public native void appendQuadratic(float xm, float ym, float x1, float y1)    throws PathError;             
              
public native void appendCubic(float xm, float ym, float xn, float yn, float x1, float y1) throws PathError;             
              
public native void closedSubpath() throws PathError;         
              
public native void endPath() throws PathError, PathException;        
              
public void useProxy(FastPathProducer proxy) throws PathError, PathException {       
proxy.sendTo(this);              
}              
              
public native long getCPathConsumer();           
              
// ___________________________________________________________________             
//              
// DEFINITION OF THE OUTPUT AREA         
// ___________________________________________________________________             
              
              
public native void getAlphaBox(int[] box) throws PRError;        
              
public native void setOutputArea(float outlox, float outloy, int w, int h)    
throws PRError, PRException;            
              
// ___________________________________________________________________             
//              
// TILE RETRIEVAL            
// ___________________________________________________________________             
              
public native int getTileState() throws PRError;         
              
public void writeAlpha(byte[] alpha, int xstride, int ystride, int pix0offset)     
throws PRError, PRException, InterruptedException {          
writeAlpha8(alpha, xstride, ystride, pix0offset);           
}              
public void writeAlpha(char[] alpha, int xstride, int ystride, int pix0offset)     
throws PRError, PRException, InterruptedException {          
writeAlpha16(alpha, xstride, ystride, pix0offset);           
}              
              
private native void writeAlpha8(byte[] alpha, int xstride, int ystride, int pix0offset) throws PRError, PRException; 
private native void writeAlpha16(char[] alpha, int xstride, int ystride, int pix0offset) throws PRError, PRException; 
              
public native void nextTile() throws PRError;         
              
public native void reset();           
              
              
// ___________________________________________________________________             
//              
// CONNECTIONS to PathFiller in C         
// ___________________________________________________________________             
static {             
java.security.AccessController.doPrivileged(              
new sun.security.action.LoadLibraryAction("dcpr"));             
cClassInitialize();              
}              
private static native void cClassInitialize();          
private static native void cClassFinalize();          
              
private long cData;            
private native void cInitialize();           
}              

