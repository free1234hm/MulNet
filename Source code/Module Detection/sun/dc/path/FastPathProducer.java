package sun.dc.path;
 
 public interface FastPathProducer {
 
    public void getBox(float[] box)     throws PathError;
 
   public void sendTo(PathConsumer pc) throws PathError, PathException;
 }
