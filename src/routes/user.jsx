import {useState} from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faEye } from '@fortawesome/free-solid-svg-icons'

function User()
{

    const [selectedImage, setSelectedImage] = useState(null);

    const handleImageChange = (event) => {
      const file = event.target.files[0];
      setSelectedImage(file);
    };
   

    return(

            <div className="main-body h-screen w-full bg-slate-100">
                    <div className="main-body-container h-screen w-full">
                    
                        <div className="bg-white mx-auto px-8 pt-6 pb-8 mb-4 w-[400px] flex flex-col justify-center overflow-clip relative top-1/2 -translate-y-1/2 shadow-md shadow-slate-300 rounded-md">

                            <h3 className="mb-6 text-left text-xl z-10">Hello! <span className="font-bold">Tom</span> <br/> Welcome to Dashboard</h3>
                                

                            <div className="dash-body flex flex-row w-full p-3 justify-between space-x-3" >


                                        <div className="upload-image text-center p-2 w-full ">

                                            <div className="upload border border-gray-300 p-10 rounded-md">
                                                <input
                                                    type="file"
                                                    accept="image/*"
                                                    onChange={handleImageChange}
                                                    className="hidden"
                                                    id="image-upload"
                                                />
                                                <label htmlFor="image-upload" className="block cursor-pointer text-[32pt]">
                                                <FontAwesomeIcon icon={faPlus} />
                                                </label>
                                                {selectedImage && (
                                                    <p className="text-green-500">{selectedImage.name}</p>
                                                )}
                                            </div>

                                            <h4 className="mt-3">Upload defect Item Image</h4>
                                        </div>

                                        <div className="Get-review text-center p-2 w-full ">

                                            <div className="review border border-gray-300 p-10 rounded-md">
                                            <label htmlFor="" className="text-[32pt]"> <FontAwesomeIcon icon={faEye} /></label>
                                              
                                            </div>

                                            <h4 className="mt-3">Track Past Orders</h4>
                                        </div>


                            </div>

                                



                               

                                
                        </div>
                    </div>
            </div>

    );
}


export default User;