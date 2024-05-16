 package sun.dc.pr;
 
 public class PRException extends java.lang.Exception
 {
 
     public static final String
         BAD_COORD_setOutputArea = "setOutputArea: alpha coordinate out of bounds",
         ALPHA_ARRAY_SHORT = "writeAlpha: alpha destination array too short",
         DUMMY = "";
 
     // Constructors
     public PRException() {
         super();
     }
 
    public PRException(String s) {
         super(s);
     }
 }
