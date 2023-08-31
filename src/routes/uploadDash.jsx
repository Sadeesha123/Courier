import {useState} from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faStarHalfStroke } from '@fortawesome/free-solid-svg-icons'
import BG from '../images/bg.jpg'
function UploadDashboard()
{

    const [selectedImage, setSelectedImage] = useState(null);

    const handleImageChange = (event) => {
      const file = event.target.files[0];
      setSelectedImage(file);
    };
   

    return(

            <div className="main-body h-screen w-full bg-slate-100">
                    <div className="main-body-container h-screen w-full">
                    
                        <div className="bg-white mx-auto  rounded px-8 pt-6 pb-8 mb-4 w-[600px] flex flex-col justify-center overflow-clip relative top-1/2 -translate-y-1/2 shadow-md shadow-slate-300 rounded-md">

                            <h3 className="mb-6 text-left text-xl z-10">Hello! <span className="font-bold">Tom</span> <br/> Welcome to Dashboard</h3>
                                

                            <div className="dash-body flex flex-row w-full p-3 justify-between space-x-3" >


                                        <div className="upload-image text-center p-2 w-full flex felx-col">

                                            <div className="upload rounded-md flex flex-row justify-center items-center space-x-5 w-full">
                                                
                                                <div className="image-holder w-2/3">
                                                                <img src={BG} alt="" srcset="" />
                                                </div>
                                                <div className="option-panel flex flex-col w-1/3 space-y-3 ">
                                                        <button className="bg-sky-200 p-2 rounded-full hover:bg-sky-400">Upload an Image</button>
                                                        <button className="bg-sky-200 p-2 rounded-full hover:bg-sky-400">Edit</button>
                                                        <button className="bg-sky-200 p-2 rounded-full hover:bg-sky-400">Remove</button>
                                                        <button className="bg-sky-200 p-2 rounded-full hover:bg-sky-400">Save for Inquiry</button>
                                                </div>
                                            </div>

                                            
                                        </div>

                                        


                            </div>

                                



                               

                                
                        </div>
                    </div>
            </div>

    );
}


export default UploadDashboard;