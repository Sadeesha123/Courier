import {useState} from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faStarHalfStroke } from '@fortawesome/free-solid-svg-icons'
import QRCode from "qrcode.react";


function QrGenerate()
{

    const [qrData, setQRData] = useState("");

    const generateQRCode = () => {
      const randomData = Math.random().toString(36).substring(7);
      setQRData(randomData);
    };

    return(

            <div className="main-body h-screen w-full bg-slate-100">
                    <div className="main-body-container h-screen w-full">
                    
                        <div className="bg-white mx-auto  px-8 pt-6 pb-8 mb-4 w-[400px] flex flex-col justify-center overflow-clip relative top-1/2 -translate-y-1/2 shadow-md shadow-slate-300 rounded-md">

                            <h3 className="mb-6 text-left text-xl z-10">Admin ID <span className="font-bold">#20CS</span> <br/></h3>
                                

                            <div className="dash-body flex flex-row w-full p-3 justify-between space-x-3" >


                                        <div className="upload-image text-center p-2 w-full ">

                                            <div className="upload border border-gray-300 p-10 rounded-md text-center justify-center items-center">
                                            {qrData ? <QRCode value={qrData} />:<p className="text-center mx-auto"></p>} 
                                            </div>

                                            
                                            <button
                                                className="mt-3 bg-purple-300 p-2 w-[130px] rounded-full text-white font-semibold"
                                                onClick={generateQRCode}
                                            >
                                                Generate QR
                                            </button>
                                        </div>

                                        


                            </div>

                                



                               

                                
                        </div>
                    </div>
            </div>

    );
}


export default QrGenerate;