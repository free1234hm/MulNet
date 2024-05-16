package sun.security.action;
 
 public class LoadLibraryAction implements java.security.PrivilegedAction {
     private String theLib;
 
     public LoadLibraryAction(String theLib) {
         this.theLib = theLib;
     }
 
     public Object run() {
         System.loadLibrary(theLib);
         return null;
     }
 }
