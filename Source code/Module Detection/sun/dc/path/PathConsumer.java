package sun.dc.path;
 
 public interface PathConsumer {
     public void beginPath() throws PathError;
 
     public void beginSubpath(float x0, float y0) throws PathError;
 
     public void appendLine(float x1, float y1) throws PathError;
 
     public void appendQuadratic(float xm, float ym,
                                 float x1, float y1) throws PathError;
 
     public void appendCubic(    float xm, float ym,
                                 float xn, float yn,
                                 float x1, float y1) throws PathError;
 
     public void closedSubpath() throws PathError;
 
     public void endPath() throws PathError, PathException;
 
     public void useProxy(FastPathProducer proxy) throws PathError, PathException;
 
     public long getCPathConsumer();
 
     public void dispose();
 
     public PathConsumer getConsumer();
 }
