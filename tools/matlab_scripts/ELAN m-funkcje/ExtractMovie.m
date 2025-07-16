function ExtractMovie(p_sInputAviFile, p_sOutputAviFile, p_oAnnotation, p_bUseCompression)

%Wycina i zapisuje na dysku twardym fragment filmu o nazwie 
%p_sInputAviFile. Wyci�ty fragment odpowiada przedzia�om czasowym 
%adnotacji p_oAnnotation. Jest on zapisywany do pliku o nazwie 
%p_sOutputAviFile. Zapisywany fragment filmu jest kompresowany 
%(kompresja Cinepak) je�li parametr p_bUseCompression ma warto�� true.
%Je�li wej�ciowy film wideo (p_sInputAviFile) jest skompresowany, 
%funkcja ta wymaga aby w systemie operacyjnym zainstalowany by� kodek,
%kt�ry potrafi go zdekompresowa�. Je�li w takiej sytuacji odpowiedni 
%kodek nie zostaje odnaleziony, funkcja zostaje przerwana z b��dem.
%
%Je�li na dysku, w kt�rym znajduje si� folder zapisu, nie ma 
%odpowiedniej ilo�ci wolnego miejsca, je�li jest on chroniony przed 
%zapisem lub je�li pliku nie mo�na zapisa� z innego powodu, funkcja 
%zostaje przerwana z b��dem.
%
%parametry:
%
%p_sInputAviFile - nazwa pliku wideo (wraz ze �cie�k�), z kt�rego ma 
%zosta� wyci�ty fragment filmu. Plik o danej nazwie musi istnie�. 
%Parametr ten musi by� typu string. W przeciwnym wypadku funkcja 
%zostaje przerwana z b��dem.
%
%p_sOutputAviFile - nazwa pliku wideo (wraz ze �cie�k�), do kt�rego ma
%zosta� zapisany wyci�ty fragment filmu. Parametr ten musi by� typu 
%string. W przeciwnym wypadku funkcja zostaje przerwana z b��dem.
%
%p_oAnnotation - obiekt adnotacji, na podstawie kt�rego wycinany jest 
%fragment filmu. Wyci�ty fragment odpowiada przedzia�om czasowym 
%adnotacji okre�lonym polami iStartIndex i iStopIndex. Parametr ten 
%musi by� obiektem struktury oAnnotation. W przeciwnym wypadku funkcja
%zostaje przerwana z b��dem.
%
%p_bUseCompression - Parametr ten okre�la, czy zapisywany fragment 
%filmu ma zosta� skompresowany. Je�li przyjmuje on warto�� true, 
%u�yta zostaje kompresja Cinepak. Wprzeciwnym wypadku, film nie jest
%kompresowany. Parametr ten musi by� typu logicznego(warto�� true lub
%false). W przeciwnym wypadku funkcja zostaje przerwana z b��dem.
%
%warto�� zwracana:
%
%brak

	if ischar(p_sInputAviFile) == 0 
		disp('Error: p_sInputAviFile is not a string.');
		return;
	end
	
	if ischar(p_sOutputAviFile) == 0 
		disp('Error: p_sOutputAviFile is not a string.');
		return;
	end

	if isstruct(p_oAnnotation) == 0
		disp('Error: p_oAnnotation is not a structure');
		return;
	end
	
	if ~isfield(p_oAnnotation, 'iStartIndex') || ~isfield(p_oAnnotation, 'iStopIndex')
		disp('Error: p_oAnnotation is a wrong structure');
		return;
	end
	
	if islogical(p_bUseCompression) == 0 
		disp('Error: p_bUseCompression is not a logical value.');
		return;
	end
	
	if p_bUseCompression
		sCompression = 'Cinepak';
	else
		sCompression = 'None';
	end
	
	try			
		sInfo = aviinfo(p_sInputAviFile);
		iFPS = sInfo.FramesPerSecond;
		
		iFirstFrame = p_oAnnotation.iStartIndex;
		iLastFrame = p_oAnnotation.iStopIndex;
		
		if iFirstFrame == 0
			iFirstFrame = 1;
		end
		
		if iLastFrame == 0
			iLastFrame = 1;
		end
		
		oInputMovie = mmreader(p_sInputAviFile);
	catch
		sTemp = strcat('Error: Cannot open a file: ''', p_sInputAviFile, '''.');
		disp(sTemp);
		return;
	end
	
	try		
		i=1;
		while iFirstFrame <= iLastFrame
			data = read(oInputMovie,iFirstFrame);
			frames(i).cdata = data;
			frames(i).colormap = [];
			
			i = i+1;
			iFirstFrame = iFirstFrame+1;
		end
	catch
		sTemp = strcat('Error: Cannot read from a file: ''', p_sInputAviFile, '''.');
		disp(sTemp);
		return;
	end
	
	try	
		oOutputMovie = avifile(p_sOutputAviFile,'compression',sCompression,'fps',iFPS);
	catch
		sTemp = strcat('Error: Cannot make a file: ''', p_sOutputAviFile, '''.');
		disp(sTemp);
		return;
	end

	try	
		for i = 1 : numel(frames)
			oOutputMovie = addframe(oOutputMovie,frames(i));
		end
	catch
		sTemp = strcat('Error: Cannot write to file: ''', p_sOutputAviFile, '''.');
		disp(sTemp);
		oOutputMovie = close(oOutputMovie);
		return;
	end
	
	oOutputMovie = close(oOutputMovie);
end






