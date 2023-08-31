import { useState, useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPlus, faStarHalfStroke } from "@fortawesome/free-solid-svg-icons";
import Sidepanel from "../../components/sidepanel";
import qr from './images/qr.jpg';
import inq from './images/inq.jpg';
import bg from '../../images/mainbg1.jpg';

function ScanQR() {
  const [stream, setStream] = useState(null);
  const videoRef = useRef(null);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((mediaStream) => {
        setStream(mediaStream);
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      })
      .catch((error) => {
        console.log("Error accessing camera:", error);
      });

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, []);




  return (
    <div className="main-body h-screen w-full bg-slate-100">
       <img src={bg} alt="" srcset="" className="object-cover w-[100%] h-[100%] fixed" />
      <div className="main-body-container h-screen w-full flex flex-row">
        <Sidepanel />
        <div className="w-5/6 side-panel p-5 md:ml-[300px] ml-16">
          <div className="common-body p-5 flex flex-col h-full items-center justify-center">
            <div className="form-body w-[80%] flex flex-row p-5 mx-auto items-center justify-center space-x-5">
              <div className="relative">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  className="w-full h-auto"
                ></video>
                
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ScanQR;
