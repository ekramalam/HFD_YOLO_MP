import os,sys,argparse
from pathlib import Path
from fall_detector_engine import FallEngine
import fall_detector_engine as _fde

def _pa():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=str,default='yolo12n.pt')
    p.add_argument('--conf',type=float,default=0.5)
    p.add_argument('--fall-threshold',type=float,default=0.1)
    p.add_argument('--angle-threshold',type=float,default=45)
    p.add_argument('--dataset-dir',type=str,default=None)
    p.add_argument('--output-dir',type=str,default='fall_snapshots')
    return p.parse_args()

def _rd(a):
    import cv2
    r=Path(a.dataset_dir)
    n=r.name
    e=FallEngine(a.model,a.conf,a.fall_threshold)
    e.angle_threshold=a.angle_threshold
    from datetime import datetime
    t=datetime.now().strftime("%Y%m%d_%H%M%S")
    po=Path("outputs");po.mkdir(parents=True,exist_ok=True)
    efn=(f"{a.output_dir}_conf{a.conf:.2f}_ft{a.fall_threshold:.2f}_at{int(a.angle_threshold)}_"
         f"AnChTh{e.angle_change_threshold:.0f}_cf{e.confirm_frames}_vdF{e.vertical_drop_frames}_"
         f"aChF{e.angle_change_frames}_{Path(a.model).stem}_{t}")
    rd=po/efn;rd.mkdir(parents=True,exist_ok=True)
    s={"total_videos":0,"adl_videos":0,"fall_videos":0,"tp":0,"tn":0,"fp":0,"fn":0}
    for v in r.rglob("*.mp4"):
        vi=v.stem
        e.current_video_name=v.name
        vs=str(v).lower()
        gf="fall" in vs
        ga="adl" in vs
        s["total_videos"]+=1
        s["fall_videos" if gf else "adl_videos"]+=1
        e.baseline_height=None
        e.prev_poses.clear()
        e.fallen_person_ids.clear()
        e.fall_detected=False
        e.video_fall_detected=False
        e.fall_start_time=None
        e.consecutive_fall_frames=0
        e.last_fall_person_id=None
        vfd=False
        fsl=False
        c=cv2.VideoCapture(str(v))
        if not c.isOpened():continue
        fps=c.get(cv2.CAP_PROP_FPS)
        fps=fps if fps>0 else 30
        e.fps=fps
        w=int(c.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rp=v.relative_to(r)
        sd=rd/rp.parent;sd.mkdir(parents=True,exist_ok=True)
        ovp=sd/f"{vi}_annotated.mp4"
        fc=cv2.VideoWriter_fourcc(*'mp4v')
        vw=cv2.VideoWriter(str(ovp),fc,fps,(w,h))
        fn=0
        while c.isOpened():
            ok,fr=c.read()
            if not ok:break
            fn+=1
            vts=fn/fps
            of,fd,fdt=e.process_frame(fr)
            if fd:
                vfd=True
                if not fsl:
                    fst=vts;fsl=True
                vd=0
                if len(e.prev_poses)>5:
                    cu=e.prev_poses[-1];pv=e.prev_poses[-6]
                ct=" | ".join(e.last_fall_criteria)
                ct+=f" | VerticalDrop={vd:.3f}"
            vw.write(of)
        vw.release();c.release()
        del fr,of
        import gc;gc.collect()
        if gf and vfd:s["tp"]+=1
        elif ga and not vfd:s["tn"]+=1
        elif ga and vfd:s["fp"]+=1
        elif gf and not vfd:s["fn"]+=1
    tp,tn,fp,fn2=s["tp"],s["tn"],s["fp"],s["fn"]


def main():
    a=_pa()
    if not os.path.exists(a.model) and not a.model.startswith('yolov12'):
        pass
    if a.dataset_dir:
        _rd(a)
    else:
        sys.exit(1)

if __name__=="__main__":
    main()