from kea import *

class Test(KeaTest):
    

    @initializer()
    def set_up(self):
        d(resourceId="net.gsantner.markor:id/next").click()
        
        d(resourceId="net.gsantner.markor:id/next").click()
        
        d(resourceId="net.gsantner.markor:id/next").click()
        
        d(resourceId="net.gsantner.markor:id/next").click()
        
        d(text="DONE").click()
        
        
        if d(text="OK").exists():
            d(text="OK").click()

    # bug #457
    @precondition(lambda self: d(resourceId="net.gsantner.markor:id/nav_notebook").exists())
    @rule()
    def swipe_should_update_title(self):
        # d.swipe_ext("left")
        if d(resourceId="net.gsantner.markor:id/nav_todo").info["selected"]:
            assert  d(resourceId="net.gsantner.markor:id/toolbar").child(text="To-Do").exists(), "To-Do not exists"
        elif d(resourceId="net.gsantner.markor:id/nav_quicknote").info["selected"]:
            assert  d(resourceId="net.gsantner.markor:id/toolbar").child(text="QuickNote").exists(), "QuickNote not exists"
        elif d(resourceId="net.gsantner.markor:id/nav_more").info["selected"]:
            assert  d(resourceId="net.gsantner.markor:id/toolbar").child(text="More").exists(),   "More not exists"




if __name__ == "__main__":
    t = Test()
    
    setting = Setting(
        apk_path="./apk/markor/2.11.1.apk",
        device_serial="emulator-5554",
        output_dir="../output/markor/457/guided_new",
        policy_name="guided"
    )
    start_kea(t,setting)
    
